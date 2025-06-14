from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
import pandas as pd
import requests


class StudentCodingScenario(Scenario):
    name = "student_coding"
    description = "Mimic student C++ style on foundational questions"
    tags = ["coding", "c++", "student"]

    def get_instances(self, output_path: str):
        df = pd.read_csv(
            "https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/sample_fifty_student_full.csv"
        )

        # Load test cases (unit tests)
        test_cases = self._load_test_cases()

        instances = []
        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                continue
            first = student_df.iloc[0]
            second = student_df.iloc[1]
            third = student_df.iloc[2]
            target = student_df.iloc[3]

            # Get test cases for this question
            question_id = target.get("question_unittest_id", None)
            question_test_cases = []
            if question_id and test_cases:
                question_test_cases = test_cases.get(str(question_id), [])
            # Get student pass (0 or 1) for the target question
            student_correctness_pattern = target.get("pass", None)
            main_part = int(student_correctness_pattern)  # "1111111111"
            # Convert each character to an int
            student_correctness_list = [int(ch) for ch in str(main_part)]  # [1,1,1,1,1,1,1,1,1,1]

            prompt = (
                f"Week: {target['week']}\n"
                f"Topic: {target['topic']}\n\n"
                "Example 1:\n"
                f"Question: {first['question_name']} — {first['question_text']}\n"
                "Template:\n"
                f"{first['question_template']}\n"
                "Your Code:\n"
                f"{first['response']}\n\n"
                "Example 2:\n"
                f"Question: {second['question_name']} — {second['question_text']}\n"
                "Template:\n"
                f"{second['question_template']}\n"
                "Your Code:\n"
                f"{second['response']}\n\n"
                "Example 3:\n"
                f"Question: {third['question_name']} — {third['question_text']}\n"
                "Template:\n"
                f"{third['question_template']}\n"
                "Your Code:\n"
                f"{third['response']}\n\n"
                "Now, using that same student style, attempt this:\n"
                f"Question: {target['question_name']} — {target['question_text']}\n"
                "Template:\n"
                f"{target['question_template']}\n\n"
                "Provide ONLY your C++ implementation following the given template, "
                "writing code just as you would in class—indentation, naming, and all."
            )
            instances.append(
                Instance(
                    id=f"{student_id}_{target['question_unittest_id']}",
                    input=Input(text=prompt),
                    references=[Reference(output=Output(text=target["response"]), tags=[CORRECT_TAG])],
                    extra_data={
                        "question_template": target["question_template"],
                        "test_cases": question_test_cases,
                        "question_id": str(question_id) if question_id else None,
                        "question_name": target.get("question_name", ""),
                        "student_id": str(student_id),
                        "student_correctness_pattern": student_correctness_list,
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
