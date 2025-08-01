from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, Output, Reference, VALID_SPLIT, CORRECT_TAG
import pandas as pd
import requests


class CodeInsightsCodeEfficiencyScenario(Scenario):
    name = "codeinsights_code_efficiency"
    description = "Evaluate runtime efficiency alignment between LLM-generated code and student code"
    tags = ["codeinsights", "c++", "code_efficiency"]

    def __init__(self, num_testcases: int = 1):
        super().__init__()
        self.num_testcases = num_testcases

    def get_instances(self, output_path: str):
        df = pd.read_csv("https://huggingface.co/datasets/Kazchoko/my_dataset/resolve/main/Scenario4_full_data.csv")

        instances = []
        skipped_no_tests = 0
        skipped_insufficient_data = 0

        for student_id, student_df in df.groupby("student_id"):
            student_df = student_df.sort_values("timestamp")
            if len(student_df) < 4:
                skipped_insufficient_data += 1
                continue

            # take exactly the first 4 attempts
            attempts = student_df.iloc[:4]

            # rotate through each as target
            for idx in range(4):
                target = attempts.iloc[idx]
                examples = [attempts.iloc[i] for i in range(4) if i != idx]

                # 1) skip if no tests available
                question_id = target.get("question_unittest_id", None)

                # 2) parse test cases
                question_test_cases = []
                tc_parsing_success = True
                for s in target["question_unittests"].split("Unittest")[1:]:
                    body = s[s.find(":") + 1 :]
                    i1, i2, i3 = body.find("Input:"), body.find("STD input:"), body.find("Output:")
                    if -1 in (i1, i2, i3):
                        tc_parsing_success = False
                        break
                    question_test_cases.append({
                        "input":  body[i1+6 : i2].strip(),
                        "std_in": body[i2+10: i3].strip(),
                        "output": body[i3+7 :].strip(),
                    })
                if not tc_parsing_success:
                    skipped_no_tests += 1
                    print(f"SKIPPING Student {student_id}, Question {question_id}: Empty test cases")
                    continue

                # 3) enforce minimum number of cases
                if len(question_test_cases) < self.num_testcases:
                    continue
                if self.num_testcases >= 0:
                    question_test_cases = question_test_cases[: self.num_testcases]

                # 4) parse correctness pattern
                pat = target.get("pass", None)
                if pat is not None:
                    main_part = int(pat)
                    student_correctness_list = [int(ch) for ch in str(main_part)]
                else:
                    student_correctness_list = []

                # log accepted instance
                print(f"\n=== ACCEPTED INSTANCE: Student {student_id}, Question {question_id} ===")
                print(f"Test cases loaded: {len(question_test_cases)}")
                print(f"Student correctness pattern: {student_correctness_list}")
                print(f"Question name: {target.get('question_name', 'MISSING')}")

                # 5) build prompt with three examples + this target
                prompt = (
                    f"Week: {target['week']}\n"
                    f"Topic: {target['topic']}\n\n"
                )
                for n, ex in enumerate(examples, start=1):
                    prompt += (
                        f"Example {n}:\n"
                        f"Question: {ex['question_name']} — {ex['question_text']}\n"
                        "Template:\n"
                        f"{ex['question_template']}\n"
                        "Your Code:\n"
                        f"{ex['response']}\n\n"
                    )
                prompt += (
                    "Now, using that same student's coding style, attempt this:\n"
                    f"Question: {target['question_name']} — {target['question_text']}\n\n"
                    + "Template:\n"
                    f"{target['question_template']}\n\n"
                    "Provide ONLY your C++ implementation that will replace the {{ STUDENT_ANSWER }} block in the template.  "
                    "– Do NOT reproduce any part of the template  "
                    "– Do NOT emit `int main()` (it’s already declared)  "
                    "– Ensure your code is correct, handles all edge cases, and includes any needed class definitions  "
                    "– Match the student’s usual efficiency style.\n\n"
                    "IMPORTANT: your entire response must be exactly one Markdown C++ code‑block:\n"
                    "1. First line: ```cpp\n"
                    "2. Last line: ```\n"
                    "No extra whitespace or text before/after.\n"
                )
                instances.append(
                    Instance(
                        id=f"{student_id}_{target['question_unittest_id']}",
                        input=Input(text=prompt),
                        references=[Reference(output=Output(text=target["response"]), tags=[CORRECT_TAG])],
                        extra_data={
                            "question_template": target["question_template"],
                            "test_cases": question_test_cases,
                            "question_id": str(question_id),
                            "question_name": target.get("question_name", ""),
                            "student_id": str(student_id),
                            "student_correctness_pattern": student_correctness_list,
                        },
                        split=VALID_SPLIT,
                    )
                )

        # Print summary statistics
        print("\n=== INSTANCE CREATION SUMMARY ===")
        print(f"Total instances created: {len(instances)}")
        print(f"Skipped (insufficient data): {skipped_insufficient_data}")
        print(f"Skipped (no test cases): {skipped_no_tests}")

        return instances
