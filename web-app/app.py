from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import os
import time

app = Flask(__name__)

JSON_DIR = "./data/"
commit_db = json.load(open(os.path.join(JSON_DIR,"CommitDatabase.json")))

tangled_buggy_methods= pd.read_csv(os.path.join(JSON_DIR,"TangledBuggyMethods.csv"))
true_buggy_methods= pd.read_csv(os.path.join(JSON_DIR,"TrueBuggyMethods.csv"))
true_not_buggy_methods= pd.read_csv(os.path.join(JSON_DIR,"TrueNotBuggyMethods.csv"))[:2000]

on_review_methods = None
method_type = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ReviewCommit')
def commitReviewer():
    project = request.args.get('project', 'view') 
    commit = request.args.get('commit', 'view') 
    return render_template('CommitReviewer.html', project=project, commit=commit)

@app.route('/ReviewMethodHistory')
def trueNotBuggyReviewer():
    method_type = request.args.get('type', 'view') 
    return render_template('MethodHistoryReviewer.html',  type=method_type)

@app.route('/ReviewTangledChanges')
def tangledReviewer():
    return render_template('TangledBuggyReviewer.html')

@app.route('/getTangledMethodList', methods=['GET'])
def getTangledMethodList():
    try:
        files = tangled_buggy_methods["Project"]+"/"+tangled_buggy_methods["File"]
        return jsonify(files.to_list())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/getMethodList', methods=['POST'])
def getMethodList():
    try:
        data = request.json
        method_type = data.get("type", "").strip()

        on_review_methods = true_buggy_methods
        if "NotBuggy" in method_type:
            on_review_methods = true_not_buggy_methods
        
        print(method_type)

        files = on_review_methods["Project"]+"/"+on_review_methods["File"]
        return jsonify(files.to_list())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/getAllChangesForMethod', methods=['POST'])
def load_method_history():
    data = request.json
    filename = data.get("filename", "").strip()
    
    project_name = filename.split("/")[0]
    file_name = filename.split("/")[1]
    
    file_path = os.path.join(JSON_DIR, "source-methods", filename)
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        json_data = json.load(open(file_path, 'r'))


        res = {"Project": project_name,
               "MethodName": json_data["functionName"],
               "Changes" : []
               }
        
        for commit_hash in json_data["changeHistoryDetails"].keys():
            item = json_data["changeHistoryDetails"][commit_hash]
            change_type = item["type"]
            if "Ymultichange" in change_type:
                item = item["subchanges"][0]
            res["Changes"].append({"CommitHash": commit_hash,
                                   "CommitMessage": item["commitMessage"],
                                   "Type": item["type"],
                                   "Source": item["actualSource"],
                                   "Diff": item["diff"]})

        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getCommitHistory', methods=['POST'])
def getCommitHistory():
    data = request.json
    project_name = data.get("project", "").strip()
    commit_hash = data.get("commit", "").strip()
    
    if not commit_hash:
        return jsonify({"error": "commit is required"}), 400
    
    try:
        res = {"Project": project_name,
                "Commit":commit_hash, 
                "CommitMessage": "",
                "Changes" : []
               }

        commit_message = ""
        print(len(commit_db[project_name][commit_hash]))
        for item in commit_db[project_name][commit_hash]:
            file_path = os.path.join(JSON_DIR, "source-methods",project_name, item["File"])
            json_data = json.load(open(file_path, 'r'))
            commit_data = json_data["changeHistoryDetails"][commit_hash]

            change_data = commit_data["subchanges"][0] if "subchanges" in commit_data.keys() else commit_data

            commit_message = change_data["commitMessage"]
            res["Changes"].append({"File": item["File"],
                                   "Type": commit_data["type"],
                                   "Source": change_data["actualSource"],
                                   "Diff": change_data["diff"]
                                   })

        res["CommitMessage"] = commit_message
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/getAllChangesForTangled', methods=['POST'])
def getAllChangesForTangled():
    data = request.json
    filename = data.get("filename", "").strip()
    
    project_name = filename.split("/")[0]
    file_name = filename.split("/")[1]
    
    file_path = os.path.join(JSON_DIR, "source-methods", filename)
    
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        json_data = json.load(open(file_path, 'r'))

        h_index = tangled_buggy_methods.loc[(tangled_buggy_methods['Project'] == project_name) & (tangled_buggy_methods['File'] == file_name)]["HistoryIndex"].to_list()[0]
        
        commit_hash = json_data["changeHistory"][h_index]

        res = {"Project": project_name, 
                "Commit":commit_hash, 
                "CommitMessage": json_data["changeHistoryDetails"][commit_hash]["commitMessage"],
                "Changes" : []
                }
        
        gold_set_path = os.path.join(JSON_DIR,"GoldSet.csv")

        if os.path.exists(gold_set_path):
            gold_set = pd.read_csv(gold_set_path)
        else:
            gold_set = None

        for item in commit_db[project_name][commit_hash]:
            file_path = os.path.join(JSON_DIR, "source-methods",project_name, item["File"])
            json_data = json.load(open(file_path, 'r'))
            commit_data = json_data["changeHistoryDetails"][commit_hash]

            change_data = commit_data["subchanges"][0] if "subchanges" in commit_data.keys() else commit_data

            decision = ""
            if gold_set is not None and gold_set.loc[(gold_set['Project'] == project_name) & (gold_set['File'] == item["File"]) & (gold_set["CommitHash"] == commit_hash)].shape[0] > 0:
                decision = gold_set.loc[(gold_set['Project'] == project_name) & (gold_set['File'] == item["File"]) & (gold_set["CommitHash"] == commit_hash)]["Decision"][0]

            res["Changes"].append({"File": item["File"],
                                   "Type": commit_data["type"],
                                   "Source": change_data["actualSource"],
                                   "Diff": change_data["diff"],
                                   "CurrentDecision": decision})

        print(res)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/submit_review', methods=['POST'])
def submit_review():
    data = request.json
    decision = data.get("decision")  # 'tick' or 'cross'
    project = data.get("project")
    file_name = data.get("file_name")
    commit_hash = data.get("commit_hash")
    diff = data.get("diff")
    
    if not all([decision, project, file_name, commit_hash]):
        return jsonify({"error": "Missing required fields"}), 400
    
    review_data = {
        "Project": project,
        "File": file_name,
        "CommitHash": commit_hash,
        "Diff": diff,
        "Decision": decision
    }

    new_df = pd.DataFrame([review_data])

    gold_set_path = os.path.join(JSON_DIR,"GoldSet.csv")

    message = "Review submitted successfully"
    if not os.path.exists(gold_set_path):
        new_df.to_csv(gold_set_path, index=False)
    else:
        df = pd.read_csv(gold_set_path)
        df.to_csv(os.path.join(JSON_DIR,"GoldSetBackup", time.strftime("%Y%m%d-%H%M%S") + "-GoldSet.csv"), index=False)
        
        if df.loc[(df['Project'] == project) & (df['File'] == file_name) & (df["CommitHash"] == commit_hash)].shape[0] == 0:
            df = pd.concat([df, new_df], ignore_index=True, sort=False)
            df.to_csv(gold_set_path, index=False)
            message = "Review submitted successfully"
        else:
            message = "This method is already reviewed."
    
    return jsonify({"message": message, "data": review_data})

@app.route('/clear_review', methods=['POST'])
def clear_review():
    data = request.json
    project = data.get("project")
    file_name = data.get("file_name")
    commit_hash = data.get("commit_hash")
    
    if not all([project, file_name, commit_hash]):
        return jsonify({"error": "Missing required fields"}), 400
    

    gold_set_path = os.path.join(JSON_DIR,"GoldSet.csv")

    df = pd.read_csv(gold_set_path)
    df.to_csv(os.path.join(JSON_DIR,"GoldSetBackup", time.strftime("%Y%m%d-%H%M%S") + "-GoldSet.csv"), index=False)
    
    df = df.drop(df[(df['Project'] == project) & (df['File'] == file_name) & (df["CommitHash"] == commit_hash)].index)
    df = df.to_csv(gold_set_path, index=False)
    
    return jsonify({"message": "Review cleared successfully"})

if __name__ == '__main__':
    app.run(debug=True)
