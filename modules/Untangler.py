from google import genai
from openai import OpenAI
import configparser
import json
import pandas as pd
import time
from tqdm import tqdm
import os
import io

random_seed = 50


class Untangler:
    def __init__(self, model_name, enable_cot=False):
        self.model_name = model_name
        self.enable_cot = enable_cot

        self.__setup()

        config = configparser.ConfigParser()
        config.read(".config")
        GEMINI_API_KEY = config["API_KEYS"]["GEMINI_API_KEY"]
        OPENAI_API_KEY = config["API_KEYS"]["OPENAI_API_KEY"]
        DEEPSEEK_API_KEY = config["API_KEYS"]["DEEPSEEK_API_KEY"]

        if self.model_name == "gemini":
            self.__client = genai.Client(api_key=GEMINI_API_KEY)
        elif self.model_name in ["openai", "deepseek"]:
            if self.model_name == "openai":
                self.__client = OpenAI(api_key=OPENAI_API_KEY)
            else:
                raise NotImplementedError()
        else:
            raise ValueError("Invalid model name")

    def __setup(self):
        self.__gemini_model_name = "gemini-2.0-flash"
        self.__openai_model_name = "gpt-4o-mini"
        self.__deepseek_model_name = ""

        self.__batch_file_name = "request.jsonl"

        self.__initial_prompt = "You are a Git commit review assistant. Given a Java source code diff and its commit message, analyze both to determine if the changes align with the described bug fix. Assess the relevance between the commit message and code modifications, identifying patterns such as error-handling updates, logical corrections, exception-handling improvements, and other indicators of bug-related changes. Use the provided examples as reference points to enhance accuracy in detecting bug-related changes."
        self.__prepare_few_shot_data()

    def __prepare_few_shot_data(self):
        few_shot_data = json.load(open("./data/FewShots.json", "r"))

        if self.model_name == "gemini":
            few_shots = "\n"
            for item in few_shot_data:
                commit_message = item["Message"].strip()
                git_diff = item["Diff"].strip()
                explanation = item["Explanation"].strip()
                answer = item["Answer"].strip()

                if self.enable_cot:
                    few_shots = (
                        few_shots
                        + f"\nCommit Messaage: {commit_message}\nGit Diff:\n{git_diff}\nAnswer: {explanation} The answer is **{answer}**.\n"
                    )
                else:
                    few_shots = (
                        few_shots
                        + f"\nCommit Messaage: {commit_message}\nGit Diff:\n{git_diff}\nAnswer: {answer}\n"
                    )

        if self.model_name == "openai":
            few_shots = []
            for item in few_shot_data:
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

    def __prepare_prompt_for_openai(self, commitMessage, diff):
        messages = [{"role": "developer", "content": self.__initial_prompt}]

        for item in self.__few_shot_data:
            messages.append(item)

        messages.append(
            {
                "role": "user",
                "content": f"Commit Messaage: {commitMessage}\nGit Diff:\n{diff}",
            }
        )
        return messages

    def prepare_batch_prompt_for_openai(self, model, messages):
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

    def __prepare_prompt_for_gemini(self, commitMessage, diff):
        question = f"\nCommit Messaage: {commitMessage}\nGit Diff:\n{diff}\nAnswer:"

        return self.__few_shot_data + question

    def prepare_prompt(self, commitMessage, diff):
        if self.model_name == "gemini":
            self.prompt = self.__prepare_prompt_for_gemini(commitMessage, diff)
        elif self.model_name == "openai":
            self.prompt = self.__prepare_prompt_for_openai(commitMessage, diff)
        else:
            raise NotImplementedError(
                f"Functionality for model: {self.model_name} has not been implemented yet."
            )

    def print_prompt(self):
        prompt = self.prompt
        if prompt == "":
            print("Printing last prompt...")
            prompt = self.last_prompt

        print("Prompt for:", self.model_name)
        print()
        print(prompt)

    def get_prompt(self):
        prompt = self.prompt
        if prompt == "":
            prompt = self.last_prompt

        return prompt

    def detect(self):
        prediction = ""
        if self.prompt == "":
            raise ValueError("Provide a new diff using prepare_prompt()")
        else:
            if self.model_name == "gemini":
                response = self.__client.models.generate_content(
                    model=self.__gemini_model_name,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=self.__initial_prompt,
                        temperature=0.3,
                    ),
                    contents=self.prompt,
                )
                prediction = response.text

            elif self.model_name == "openai":
                completion = self.__client.chat.completions.create(
                    model=self.__openai_model_name,
                    messages=self.prompt,
                    temperature=0.3,
                )
                prediction = completion.choices[0].message.content

            elif self.model_name == "deepseek":
                raise NotImplementedError()

            self.last_prompt = self.prompt
            self.prompt = ""

            return prediction

    def batch_detect(self, df):
        y_pred = []
        y_true = []
        for index, row in df.iterrows():
            y_true.append(row["Decision"])

        if self.model_name == "gemini":
            for index, row in tqdm(df.iterrows()):
                error = True
                self.prepare_prompt(row["CommitMessage"], row["Diff"])
                while error:
                    try:
                        pred = self.predict()
                        error = False
                    except genai.errors.ClientError:
                        error = True
                        time.sleep(60)
                y_pred.append(pred)
                y_true.append(row["Decision"])
                time.sleep(3)

        elif self.model_name == "openai":

            messages = []
            for index, row in tqdm(df.iterrows()):
                messages.append(
                    self.__prepare_prompt_for_openai(row["CommitMessage"], row["Diff"])
                )

            self.prepare_batch_prompt_for_openai(self.__openai_model_name, messages)

            batch_input_file = self.__client.files.create(
                file=open(self.__batch_file_name, "rb"), purpose="batch"
            )

            batch_input_file_id = batch_input_file.id
            self.__current_openai_batch = self.__client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            self.__current_openai_batch, y_pred = self.get_batch_result()
        else:
            raise NotImplementedError()

        return y_true, y_pred

    def get_batch_result(self, batch_id=None):
        if batch_id == None:
            batch_id = self.__current_openai_batch.id

        batch = self.__client.batches.retrieve(batch_id)

        with tqdm(total=batch.request_counts.total) as pbar:
            pbar.update(batch.request_counts.completed)
            while batch.status != "completed":
                time.sleep(5)
                batch = self.__client.batches.retrieve(batch_id)
                pbar.update(max(0, batch.request_counts.completed - pbar.n))

        y_pred = []
        if batch.status == "completed":
            file_response = self.__client.files.content(batch.output_file_id)
            jsonObj = pd.read_json(io.StringIO(file_response.text), lines=True)
            for r in jsonObj["response"].to_list():
                y_pred.append(r["body"]["choices"][0]["message"]["content"])
        return batch, y_pred
