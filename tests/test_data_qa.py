import os
import sys
import unittest
from unittest.mock import patch


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from d5p4.config import Config
from d5p4.data.qa import get_qa_dataset


class _FakeSplit(list):
    def shuffle(self, seed: int):
        del seed
        return _FakeSplit(self)


class TestQaDatasets(unittest.TestCase):
    @patch("d5p4.data.qa.load_dataset")
    def test_commonsense_qa_uses_validation_split(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            "validation": _FakeSplit(
                [
                    {
                        "question": "Where would a hooker give services?",
                        "choices": {"text": ["garage", "street corner", "library"], "label": ["A", "B", "C"]},
                        "answerKey": "B",
                    },
                ],
            ),
            "test": _FakeSplit(
                [
                    {
                        "question": "Where would a hooker give services?",
                        "choices": {"text": ["garage", "street corner", "library"], "label": ["A", "B", "C"]},
                        "answerKey": "",
                    },
                ],
            ),
            "train": _FakeSplit([]),
        }

        cfg = Config(disable_sys_args=True, qa_dataset="commonsense_qa")

        df = get_qa_dataset(cfg)

        self.assertEqual(df.loc[0, "correct_answers"], ["street corner"])
        self.assertEqual(df.loc[0, "incorrect_answers"], ["garage", "library"])
        mock_load_dataset.assert_called_once_with("tau/commonsense_qa", cache_dir="./.cache")

    @patch("d5p4.data.qa.load_dataset")
    def test_commonsense_qa_falls_back_to_default_dataset_id(self, mock_load_dataset):
        mock_load_dataset.side_effect = [
            FileNotFoundError("stale cluster path"),
            {
                "validation": _FakeSplit(
                    [
                        {
                            "question": "Where do people usually keep books?",
                            "choices": {"text": ["library", "garage", "beach"], "label": ["A", "B", "C"]},
                            "answerKey": "A",
                        },
                    ],
                ),
                "train": _FakeSplit([]),
            },
        ]

        cfg = Config(
            disable_sys_args=True,
            qa_dataset="commonsense_qa",
            commonsense_qa_path="/missing/commonsense_qa",
        )

        df = get_qa_dataset(cfg)

        self.assertEqual(df.loc[0, "correct_answers"], ["library"])
        self.assertEqual(mock_load_dataset.call_args_list[0].args, ("/missing/commonsense_qa",))
        self.assertEqual(mock_load_dataset.call_args_list[1].args, ("tau/commonsense_qa",))

    @patch("d5p4.data.qa.load_dataset")
    def test_ai2_arc_formats_correct_and_incorrect_answers(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            "test": _FakeSplit(
                [
                    {
                        "question": "Which planet is known as the Red Planet?",
                        "choices": {"text": ["Earth", "Mars", "Venus"], "label": ["A", "B", "C"]},
                        "answerKey": "B",
                    },
                ],
            ),
            "train": _FakeSplit([]),
        }

        cfg = Config(disable_sys_args=True, qa_dataset="ai2_arc")

        df = get_qa_dataset(cfg)

        self.assertEqual(
            df.to_dict("records"),
            [
                {
                    "question": "Which planet is known as the Red Planet?",
                    "correct_answers": ["Mars"],
                    "incorrect_answers": ["Earth", "Venus"],
                },
            ],
        )
        mock_load_dataset.assert_called_once_with("allenai/ai2_arc", "ARC-Challenge", cache_dir="./.cache")

    @patch("d5p4.data.qa.load_dataset")
    def test_ai2_arc_few_shot_prefix_uses_choice_labels(self, mock_load_dataset):
        mock_load_dataset.return_value = {
            "test": _FakeSplit(
                [
                    {
                        "question": "What gas do plants absorb from the atmosphere?",
                        "choices": {
                            "text": ["Oxygen", "Carbon dioxide", "Nitrogen"],
                            "label": ["1", "2", "3"],
                        },
                        "answerKey": "2",
                    },
                ],
            ),
            "train": _FakeSplit(
                [
                    {
                        "question": "What force keeps planets in orbit around the sun?",
                        "choices": {"text": ["Magnetism", "Gravity", "Friction"], "label": ["x", "y", "z"]},
                        "answerKey": "y",
                    },
                ],
            ),
        }

        cfg = Config(disable_sys_args=True, qa_dataset="ai2_arc", qa_n_shots=1)

        df = get_qa_dataset(cfg)

        self.assertEqual(
            df.loc[0, "question"],
            "Question: What force keeps planets in orbit around the sun?\n"
            "Answer: Gravity\n\n"
            "Question: What gas do plants absorb from the atmosphere?\n"
            "Answer:",
        )
        self.assertEqual(df.loc[0, "correct_answers"], ["Carbon dioxide"])
        self.assertEqual(df.loc[0, "incorrect_answers"], ["Oxygen", "Nitrogen"])


if __name__ == "__main__":
    unittest.main()
