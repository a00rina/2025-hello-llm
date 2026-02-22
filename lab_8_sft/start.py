"""
Fine-tuning starter.
"""

from pathlib import Path

from core_utils.llm.metrics import Metrics
from core_utils.project.lab_settings import LabSettings
from lab_7_llm.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    report_time,
    TaskDataset,
    TaskEvaluator,
)
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    model = BertForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-toxicity")
    print(model)
    current_path = Path(__file__).parent
    settings = LabSettings(current_path / "settings.json")
    
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    
    if importer.raw_data is None:
        raise ValueError(
            '''"RawDataPreprocessor" has incompatible type
                          "DataFrame | None"; expected "DataFrame"'''
        )
    
    preprocessor = RawDataPreprocessor(importer.raw_data)
    preprocessor.transform()
    
    dataset = TaskDataset(preprocessor.data.head(100))
    pipeline = LLMPipeline(
        model_name=settings.parameters.model,
        dataset=dataset,
        max_length=120,
        batch_size=1,
        device="cpu",
    )
    
    model_properties = pipeline.analyze_model()
    for k, v in model_properties.items():
        print(f"{k}: {v}")
    
    print(pipeline.infer_sample(dataset[0]))
    
    # assert result is not None, "Fine-tuning does not work correctly"


if __name__ == "__main__":
    main()
