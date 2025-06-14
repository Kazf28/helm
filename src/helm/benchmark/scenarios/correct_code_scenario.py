from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, VALID_SPLIT
import pandas as pd
import requests


class CorrectCodeScenario(Scenario):
    name = "correct_code"
    description = "Generate correct response code for C++ programming questions"
    tags = ["coding", "c++", "student"]

    def get_instances(self, output_path: str):
        df = pd.read_csv(
            "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/sample_fifty_student_full.csv"
        )

        # Load test cases (unit tests)
        test_cases = self._load_test_cases()

        instances = []
        for question_id, question_df in df.groupby("question_unittest_id"):
            target = question_df.iloc[0]
            question_test_cases = []
            if question_id and test_cases:
                question_test_cases = test_cases.get(str(question_id), [])

            prompt = (
                f"Question: {target['question_name']} — {target['question_text']}\n\n"
                "Template:\n"
                f"{target['question_template']}\n\n"
                "Provide ONLY your C++ implementation following the given template. "
                "Ensure your code is correct, efficient, and handles all edge cases properly."
            )
            instances.append(
                Instance(
                    id=f"{question_id}",
                    input=Input(text=prompt),
                    references=[],
                    extra_data={
                        "question_template": target["question_template"],
                        "test_cases": question_test_cases,
                        "question_id": str(question_id) if question_id else None,
                        "question_name": target.get("question_name", ""),
                    },
                    split=VALID_SPLIT,
                )
            )
        return instances

    def _load_test_cases(self):
        """
        Load test cases from external source or return None if not available.
        This method should be implemented based on where your test cases are stored.

        Expected format:
        {
            "question_id": [
                {
                    "unittest": "test_id",
                    "input": "test input code",
                    "output": "expected output"
                },
                ...
            ],
            ...
        }
        """
        try:
            response = requests.get(
                "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/test_cases_by_qid.json"
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Failed to load test cases from URL: {e}")
            return {}
