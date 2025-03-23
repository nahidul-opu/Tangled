from google import genai
from openai import OpenAI
import configparser
import json
import pandas as pd
import time
from tqdm import tqdm
import os
import io
import re
import tiktoken


class BaseUntangler:
    def __init__(self, model_name="", shot_count=0, enable_cot=False):
        if model_name == "":
            raise ValueError("Model name is required")

        self.model_name = model_name
        self.shot_count = shot_count
        self.enable_cot = enable_cot

        self.__setup_prompts()

    def __setup_prompts(self):
        persona = "You are a Git commit review assistant. You excel at analyzing Java source code diffs and commit messages."
        instruction = "Given a Java source code diff and its commit message, analyze both to determine if the changes align with the described bug fix."
        context = "Assess the relevance between the commit message and code modifications, identifying patterns such as error-handling updates, logical corrections, exception-handling improvements, and other indicators of bug-related changes."

        if self.enable_cot:
            format = "Think step by step, exaplain the reasoning, and provide an answer as **Buggy** if the changes align; otherwise, answer with **NotBuggy**."
        else:
            format = "The output should be a single word. If the changes align, output 'Buggy'; otherwise, output 'NotBuggy'."

        self.initial_prompt = f"{persona} {instruction} {context} {format}"

    def get_token_count(self, prompt):
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            total_tokens = 0
            
            for message in prompt:
                content = message.get("content", "")
                total_tokens += len(encoding.encode(content))
            
            return total_tokens
        except:
            return "<Could not count tokens>"

    def get_prompt(self):
        prompt = self.prompt
        if prompt == "":
            prompt = self.last_prompt

        return prompt

    def print_prompt(self):
        prompt = self.prompt
        if prompt == "":
            print("Printing last prompt...")
            prompt = self.last_prompt

        print("Prompt for:", self.model_name)
        print("Token count:", self.get_token_count(prompt))
        print()
        print(prompt)

    def extract_cot_based_result(self, result):
        def extract_last_sentence(text):
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            return sentences[-1] if sentences else ""

        def extract_bold_buggy(text):
            bold_buggy_pattern = r"\*\*(.*?Buggy.*?)\*\*"
            bold_buggy_words = re.findall(bold_buggy_pattern, text)
            if len(bold_buggy_words) == 0:
                return text
            else:
                return bold_buggy_words[0]

        answer = extract_last_sentence(result)
        answer = extract_bold_buggy(answer)

        return result, answer


class GeminiUntangler(BaseUntangler):
    def __init__(self, model_name="gemini-2.0-flash", shot_count=0, enable_cot=False):
        super().__init__(model_name, shot_count, enable_cot)
        self.__setup()

    def __setup(self):
        config = configparser.ConfigParser()
        config.read(".config")
        GEMINI_API_KEY = config["API_KEYS"]["GEMINI_API_KEY"]

        self.__client = genai.Client(api_key=GEMINI_API_KEY)

        self.__prepare_few_shot_data()

    def __prepare_few_shot_data(self):
        if self.shot_count == 0:
            self.__few_shot_data = ""
        else:
            few_shot_data = json.load(open("./data/FewShots.json", "r"))
            few_shots = "<EXAMPLE>"

            for indx, item in enumerate(few_shot_data):
                if indx == self.shot_count:
                    break
                commit_message = item["Message"].strip()
                git_diff = item["Diff"].strip()
                explanation = item["Explanation"].strip()
                answer = item["Answer"].strip()

                if self.enable_cot:
                    few_shots += f"\nCommit Messaage: {commit_message}\nGit Diff:\n{git_diff}\nAnswer: {explanation} The answer is **{answer}**.\n"
                else:
                    few_shots += f"\nCommit Messaage: {commit_message}\nGit Diff:\n{git_diff}\nAnswer: {answer}\n"

            self.__few_shot_data = few_shots + "</EXAMPLE>\n"

    def prepare_prompt(self, commitMessage, diff):
        question = f"\nCommit Messaage: {commitMessage}\nGit Diff:\n{diff}\nAnswer:"
        self.prompt = self.__few_shot_data + question

    def detect(self):
        if self.prompt == "":
            raise ValueError("Provide a new diff using prepare_prompt()")

        response = self.__client.models.generate_content(
            model=self.model_name,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.initial_prompt,
                temperature=0.3,
            ),
            contents=self.prompt,
        )
        prediction = response.text

        if self.enable_cot:
            return self.extract_cot_based_result(prediction)

        return prediction.strip()

    def batch_detect(self, df):
        df = df.copy()

        explanations = []
        answers = []

        for index, row in tqdm(df.iterrows()):
            error = True
            self.prepare_prompt(row["CommitMessage"], row["Diff"])

            while error:
                try:
                    pred = self.detect()
                    error = False
                except genai.errors.ClientError:
                    error = True
                    time.sleep(10)

            if self.enable_cot:
                e, a = self.extract_cot_based_result(pred)
                explanations.append(e)
                answers.append(a)
            else:
                explanations.append("")
                answers.append(pred)

            #time.sleep(4)

        df["Detection"] = answers
        if self.enable_cot:
            df["Explanation"] = explanations
        
        return df


class OpenAIUntangler(BaseUntangler):
    def __init__(self, model_name="gpt-4o-mini", shot_count=0, enable_cot=False):
        super().__init__(model_name, shot_count, enable_cot)
        self.__setup()

    def __setup(self):
        config = configparser.ConfigParser()
        config.read(".config")
        OPENAI_API_KEY = config["API_KEYS"]["OPENAI_API_KEY"]

        self.__client = OpenAI(api_key=OPENAI_API_KEY)

        self.__batch_file_name = "request.jsonl"

        self.__prepare_few_shot_data()

    def __prepare_few_shot_data(self):
        if self.shot_count == 0:
            self.__few_shot_data = []
        else:
            few_shot_data = json.load(open("./data/FewShots.json", "r"))
            few_shots = []

            for indx, item in enumerate(few_shot_data):
                if indx == self.shot_count:
                    break
                commit_message = item["Message"].strip()
                git_diff = item["Diff"].strip()
                explanation = item["Explanation"].strip()
                answer = item["Answer"].strip()

                few_shots.append(
                    {
                        "role": "user",
                        "content": f"Commit Messaage: {commit_message}\nGit Diff:\n{git_diff}",
                    }
                )

                if self.enable_cot:
                    few_shots.append(
                        {
                            "role": "assistant",
                            "content": f"{explanation} The answer is **{answer}**.",
                        }
                    )
                else:
                    few_shots.append({"role": "assistant", "content": answer})

            self.__few_shot_data = few_shots

    def prepare_prompt(self, commitMessage, diff):
        messages = [{"role": "developer", "content": self.initial_prompt}]
        for item in self.__few_shot_data:
            messages.append(item)
        messages.append(
            {
                "role": "user",
                "content": f"Commit Messaage: {commitMessage}\nGit Diff:\n{diff}",
            }
        )
        self.prompt = messages

    def __prepare_batch_prompt(self, model, messages):
        if os.path.exists(self.__batch_file_name):
            os.remove(self.__batch_file_name)

        jsonl_file = open(self.__batch_file_name, "a")

        for i, message in enumerate(messages):
            request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model, "messages": message},
            }
            jsonl_file.write(json.dumps(request) + "\n")
        jsonl_file.close()

    def detect(self):
        if self.prompt == "":
            raise ValueError("Provide a new diff using prepare_prompt()")
        completion = self.__client.chat.completions.create(
            model=self.model_name,
            messages=self.prompt,
            temperature=0.3,
        )
        prediction = completion.choices[0].message.content

        if self.enable_cot:
            return self.extract_cot_based_result(prediction)

        return prediction.strip()

    def batch_detect(self, df, iteratively=False):
        df = df.copy()

        if iteratively:
            explanations = []
            answers = []
            for index, row in tqdm(df.iterrows()):
                error = True
                self.prepare_prompt(row["CommitMessage"], row["Diff"])
                while error:
                    try:
                        pred = self.detect()
                        error = False
                    except Exception as e:
                        error = True
                        print(f"Error - {e}.\nRetrying in 0.1 minute")
                        time.sleep(10)
                if self.enable_cot:
                    e, a = self.extract_cot_based_result(pred)
                    explanations.append(e)
                    answers.append(a)
                else:
                    explanations.append("")
                    answers.append(pred)
                #time.sleep(2.5)

            df["Detection"] = answers
            if self.enable_cot:
                df["Explanation"] = explanations
        else:
            messages = []
            for index, row in tqdm(df.iterrows()):
                self.prepare_prompt(row["CommitMessage"], row["Diff"])
                messages.append(self.prompt)
            self.__prepare_batch_prompt(self.model_name, messages)
            batch_input_file = self.__client.files.create(
                file=open(self.__batch_file_name, "rb"), purpose="batch"
            )
            batch_input_file_id = batch_input_file.id
            self.__current_openai_batch = self.__client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            self.__current_openai_batch, df = self.get_batch_result(df=df)

        return df

    def get_batch_result(self, batch_id=None, df=None):
        if batch_id == None:
            batch_id = self.__current_openai_batch.id

        batch = self.__client.batches.retrieve(batch_id)

        with tqdm(total=batch.request_counts.total) as pbar:
            pbar.update(batch.request_counts.completed)
            while batch.status != "completed":
                time.sleep(5)
                batch = self.__client.batches.retrieve(batch_id)
                pbar.update(max(0, batch.request_counts.completed - pbar.n))
                if batch.status in ["failed", "expired"]:
                    raise Exception(batch.errors)

        explanations = []
        answers = []
        if batch.status == "completed":
            file_response = self.__client.files.content(batch.output_file_id)
            jsonObj = pd.read_json(io.StringIO(file_response.text), lines=True)
            for r in jsonObj["response"].to_list():
                pred = r["body"]["choices"][0]["message"]["content"]
                if self.enable_cot:
                    e, a = self.extract_cot_based_result(pred)
                    explanations.append(e)
                    answers.append(a)
                else:
                    explanations.append("")
                    answers.append(pred)

        if df is not None:
            df = df.copy()
            df["Detection"] = answers
            if self.enable_cot:
                df["Explanation"] = explanations
        else:
            if self.enable_cot:
                df = pd.DataFrame(
                    [explanations, answers], columns=["Explanation", "Detection"]
                )
            else:
                df = pd.DataFrame(
                    [answers], columns=["Detection"]
                )
        return batch, df


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class FreeUntangler(BaseUntangler):
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", shot_count=0, enable_cot=False):
        super().__init__(model_name, shot_count, enable_cot)
        self.__setup()

    def __setup(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=500,
            do_sample=False,
        )

    def __prepare_few_shot_data(self):
        if self.shot_count == 0:
            self.__few_shot_data = []
        else:
            few_shot_data = json.load(open("./data/FewShots.json", "r"))
            few_shots = []

            for indx, item in enumerate(few_shot_data):
                if indx == self.shot_count:
                    break
                commit_message = item["Message"].strip()
                git_diff = item["Diff"].strip()
                explanation = item["Explanation"].strip()
                answer = item["Answer"].strip()

                few_shots.append(
                    {
                        "role": "user",
                        "content": f"Commit Messaage: {commit_message}\nGit Diff:\n{git_diff}",
                    }
                )

                if self.enable_cot:
                    few_shots.append(
                        {
                            "role": "assistant",
                            "content": f"{explanation} The answer is **{answer}**.",
                        }
                    )
                else:
                    few_shots.append({"role": "assistant", "content": answer})

            self.__few_shot_data = few_shots

    def prepare_prompt(self, commitMessage, diff):
        messages = [{"role": "user", "content": self.initial_prompt}]
        for item in self.__few_shot_data:
            messages.append(item)
        messages.append(
            {
                "role": "user",
                "content": f"Commit Messaage: {commitMessage}\nGit Diff:\n{diff}",
            }
        )
        self.prompt = messages

    def detect(self):
        if self.prompt == "":
            raise ValueError("Provide a new diff using prepare_prompt()")
        output = self.pipe(self.prompt)
        prediction = output[0]["generated_text"]

        if self.enable_cot:
            return self.extract_cot_based_result(prediction)

        return prediction.strip()

    def batch_detect(self, df):
        df = df.copy()

        explanations = []
        answers = []

        for index, row in tqdm(df.iterrows()):
            error = True
            self.prepare_prompt(row["CommitMessage"], row["Diff"])

            while error:
                try:
                    pred = self.detect()
                    error = False
                except Exception as e:
                    error = True
                    print(f"Error - {e}.\nRetrying...")
                    time.sleep(1)

            if self.enable_cot:
                e, a = self.extract_cot_based_result(pred)
                explanations.append(e)
                answers.append(a)
            else:
                explanations.append("")
                answers.append(pred)

        df["Detection"] = answers
        if self.enable_cot:
            df["Explanation"] = explanations

        return df