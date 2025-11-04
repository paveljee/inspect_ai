from enum import Enum
from typing import Any, List, Literal, Optional, overload

from pydantic import BaseModel

# ruff: noqa: F841, E402


class CheatsheetParser:
    import os

    PDF_PATH = os.getenv("CHEATSHEET_PDF_PATH")
    FLOW_DIAGRAM_PDF_PATH = os.getenv("CHEATSHEET_FLOW_DIAGRAM_PDF_PATH")

    class MetaSchema(BaseModel):
        class ExtractionField(BaseModel):
            """Represents a single field to be extracted"""

            class FieldType(Enum):
                TEXT = "text"
                BOOLEAN = "boolean"
                CATEGORICAL = "categorical"
                NUMBER = "number"
                DATE = "date"

            class QuestionType(Enum):
                SCREENING = "screening"  # Regular screening question
                FLAGGING = "flagging"  # Organizational/flagging question

            name: str
            description: str
            field_type: FieldType
            required: bool = True
            question_number: Optional[int] = None
            question_type: QuestionType = QuestionType.SCREENING

            # For categorical fields
            allowed_values: Optional[List[str]] = None

            # Guidance for extraction
            guidance_notes: Optional[str] = None

            # Logic that affects other fields/questions
            conditional_logic: Optional[str] = None

        class FlatDict(BaseModel):
            """'Something utterly disgusting' - see here: <https://github.com/googleapis/python-genai/issues/460>"""

            class Item(BaseModel):
                key: str
                value: str

            items: List[Item]

            def to_dict(self) -> dict:
                return {item.key: item.value for item in self.items}

            @classmethod
            def from_dict(cls, d: dict[Any, Any]):
                """Note: Flattens dict by stringifying all keys and values."""
                items = [cls.Item(key=str(k), value=str(v)) for k, v in d.items()]
                return cls(items=items)

        """Complete extraction specification for a document type"""
        document_type: str
        purpose: str
        fields: List[ExtractionField]

        # Document-level context
        review_objective: Optional[str] = None
        review_questions: Optional[List[str]] = None
        project_details: Optional[FlatDict] = None

        # Global processing rules
        global_instructions: Optional[List[str]] = None

        # Reference definitions (like footnotes)
        definitions: Optional[FlatDict] = None

        @overload
        def build_extraction_schema(
            self, *, per_question: Literal[False] = False
        ) -> type[BaseModel]: ...

        @overload
        def build_extraction_schema(
            self, *, per_question: Literal[True]
        ) -> List[type[BaseModel]]: ...

        def build_extraction_schema(self, *, per_question: bool = False):
            """Convert MetaSchema instance to one or multiple Pydantic model classes for extraction"""
            from pydantic import create_model

            # The below are defined simply as a shortcut
            MetaSchema = type(self)
            ExtractionField = MetaSchema.ExtractionField
            FieldType = ExtractionField.FieldType
            QuestionType = ExtractionField.QuestionType  # <rdf:Description rdf:about="tag:pzhelnov@p1m.org,2025-07-31:AICODE-NOTE"><skos:note>unused - no special behavior/logic needs; but keeping the variable for consistency</skos:note></rdf:Description>
            FlatDict = MetaSchema.FlatDict

            # Create FlatDict class
            class FlatDictLiteral(FlatDict):
                pass

            # Create individual ExtractionFieldLiteral classes for each field
            extraction_field_classes = {}
            extracted_value_classes = {}

            for i, field in enumerate(self.fields, 1):
                # Get the actual field names and structure from the ExtractionField class
                field_attrs = {}

                # Iterate through all fields defined in the ExtractionField class
                for field_name, field_info in ExtractionField.model_fields.items():
                    field_value = getattr(field, field_name)

                    if field_value is not None:
                        # Handle different field types appropriately
                        # Also stringify all of them for Gemini compliance
                        if isinstance(field_value, Enum):
                            field_attrs[field_name] = Literal[str(field_value.value)]
                        elif isinstance(field_value, list):
                            # Convert list to tuple for Literal
                            field_attrs[field_name] = Literal[
                                tuple([str(item) for item in field_value])
                            ]
                        elif isinstance(field_value, (bool, int, float)):
                            # Short enough - just stringify for Gemini
                            field_attrs[field_name] = Literal[str(field_value)]
                        else:
                            # Regular field - use the actual value instead of hash
                            # The hash approach was causing models to use hash values instead of real content
                            field_attrs[field_name] = Literal[str(field_value)]

                # Create the field-specific ExtractionField class
                ExtractionFieldClass = create_model(
                    f"ExtractionField{i}Literal", **field_attrs
                )
                extraction_field_classes[i] = ExtractionFieldClass

                # Determine the appropriate type for extracted values based on field_type
                if field.field_type == FieldType.CATEGORICAL:
                    if field.allowed_values:
                        value_type = Literal[tuple(field.allowed_values)]
                    else:
                        value_type = str
                elif field.field_type == FieldType.TEXT:
                    value_type = str
                elif field.field_type == FieldType.BOOLEAN:
                    value_type = bool
                elif field.field_type == FieldType.NUMBER:
                    value_type = float
                elif field.field_type == FieldType.DATE:
                    value_type = str
                else:
                    value_type = str

                # Create field-specific ExtractedValue class
                extracted_value_attrs = {
                    "step_0_extraction_field": (ExtractionFieldClass, ...),
                    "step_1_initial_reflection_on_extracted_value": (str, ...),
                    "step_2_initial_extracted_value": (value_type, ...),
                    "step_3_peer_reviewer_1_critical_comments_on_initial_extracted_value": (
                        List[str],
                        ...,
                    ),
                    "step_4_peer_reviewer_2_critical_comments_on_initial_extracted_value": (
                        List[str],
                        ...,
                    ),
                    "step_5_peer_reviewer_3_critical_comments_on_initial_extracted_value": (
                        List[str],
                        ...,
                    ),
                    "step_6_response_to_peer_reviewers_comments": (str, ...),
                    "step_7_final_extracted_value": (value_type, ...),
                }

                ExtractedValueClass = create_model(
                    f"ExtractedValue{i}", **extracted_value_attrs
                )
                extracted_value_classes[i] = ExtractedValueClass

            # Create ExtractedValues class that contains all field_X properties
            extracted_values_attrs = {}
            for i in range(1, len(self.fields) + 1):
                extracted_values_attrs[f"field_{i}"] = (extracted_value_classes[i], ...)
            ExtractedValues = create_model("ExtractedValues", **extracted_values_attrs)

            # Build field definitions for the main ExtractionSchema
            base_field_definitions = {}

            # Dynamically read all fields from the MetaSchema class
            for field_name, field_info in MetaSchema.model_fields.items():
                if field_name == "fields":
                    continue

                field_value = getattr(self, field_name)

                if field_value is not None:
                    # Handle different field types appropriately
                    if isinstance(field_value, list):
                        # Convert list to tuple for Literal
                        base_field_definitions[field_name] = Literal[
                            tuple([str(item) for item in field_value])
                        ]
                    elif hasattr(type(field_value), "model_fields"):
                        if isinstance(field_value, FlatDict):
                            # This is a nested model (like FlatDict), use the Literal version
                            base_field_definitions[field_name] = Optional[
                                FlatDictLiteral
                            ]
                        else:
                            # Don't know how to handle other cases
                            base_field_definitions[field_name] = Literal[
                                str(field_value)
                            ]
                    elif isinstance(field_value, (bool, int, float)):
                        # Short enough - just stringify for Gemini
                        base_field_definitions[field_name] = Literal[str(field_value)]
                    else:
                        # Regular field - use the actual value instead of hash
                        # The hash approach was causing models to use hash values instead of real content
                        base_field_definitions[field_name] = Literal[str(field_value)]
                else:
                    # Field is None, make it optional with appropriate type
                    if field_info.annotation and hasattr(
                        field_info.annotation, "__origin__"
                    ):
                        # It's already Optional or List, keep the structure but make it optional
                        if "List" in str(field_info.annotation):
                            base_field_definitions[field_name] = Optional[List[str]]
                        elif "FlatDict" in str(field_info.annotation):
                            base_field_definitions[field_name] = Optional[
                                FlatDictLiteral
                            ]
                        else:
                            base_field_definitions[field_name] = Optional[str]
                    else:
                        base_field_definitions[field_name] = Optional[str]

            def calculate_relevance_score(extraction_schema_instance) -> float:
                """Calculate relevance score based on step_7_final_extracted_value fields"""
                for field_attr in extraction_schema_instance.values.__dict__.values():
                    final_value = getattr(
                        field_attr, "step_7_final_extracted_value", None
                    )
                    if isinstance(final_value, str) and final_value.lower() == "no":
                        return 0.0
                    elif isinstance(final_value, bool) and final_value is False:
                        return 0.0
                return 1.0

            def build_schema(
                name: str,
                extracted_values_model: type[BaseModel],
                field_index: int | None = None,
            ):
                field_definitions = dict(base_field_definitions)
                field_definitions["values"] = (extracted_values_model, ...)
                schema = create_model(name, **field_definitions)

                if field_index is None:
                    for i, extraction_field_class in extraction_field_classes.items():
                        setattr(
                            schema, f"ExtractionField{i}Literal", extraction_field_class
                        )
                        setattr(
                            schema, f"ExtractedValue{i}", extracted_value_classes[i]
                        )
                else:
                    setattr(
                        schema,
                        "ExtractionFieldLiteral",
                        extraction_field_classes[field_index],
                    )
                    setattr(
                        schema, "ExtractedValue", extracted_value_classes[field_index]
                    )

                setattr(schema, "ExtractedValues", extracted_values_model)
                schema.FlatDict = FlatDictLiteral
                schema.calculate_relevance_score = staticmethod(
                    calculate_relevance_score
                )
                return schema

            ExtractionSchema = build_schema("ExtractionSchema", ExtractedValues)

            if per_question:
                per_question_schemas: List[type[BaseModel]] = []
                for i in range(1, len(self.fields) + 1):
                    extracted_values_model = create_model(
                        "ExtractedValuesField",
                        **{"field": (extracted_value_classes[i], ...)},
                    )
                    per_question_schemas.append(
                        build_schema(
                            "ExtractionSchemaField",
                            extracted_values_model,
                            field_index=i,
                        )
                    )
                return per_question_schemas

            return ExtractionSchema
