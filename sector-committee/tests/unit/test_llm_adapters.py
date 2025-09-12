"""Unit tests for LLM adapters module.

This module tests the sector-specific adapters that bridge pure LLM models
with scoring logic, including the adapter factory and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from sector_committee.data_models import SectorRequest, ModelResult
from sector_committee.llm_models import ModelName
from sector_committee.llm_models import (
    LLMModel,
    LLMError,
    StandardLLMModel,
    TwoStageLLMModel,
    LLMResponse,
)
from sector_committee.scoring.llm_adapters import (
    SectorAnalysisAdapter,
    StandardSectorAdapter,
    DeepResearchSectorAdapter,
    SectorAnalysisError,
    SectorAdapterFactory,
)


class TestSectorAnalysisError:
    """Test SectorAnalysisError exception class."""

    def test_error_creation_basic(self):
        """Test basic error creation."""
        error = SectorAnalysisError(ModelName.OPENAI_GPT4, "Test error message")

        assert error.model == ModelName.OPENAI_GPT4
        assert error.original_error is None
        assert str(error) == "gpt-4: Test error message"

    def test_error_creation_with_original(self):
        """Test error creation with original exception."""
        original = ValueError("Original error")
        error = SectorAnalysisError(ModelName.OPENAI_GPT4O, "Wrapper error", original)

        assert error.model == ModelName.OPENAI_GPT4O
        assert error.original_error == original
        assert str(error) == "gpt-4o: Wrapper error"

    def test_error_inheritance(self):
        """Test that SectorAnalysisError inherits from Exception."""
        error = SectorAnalysisError(ModelName.OPENAI_GPT4, "Test")
        assert isinstance(error, Exception)


class TestStandardSectorAdapter:
    """Test StandardSectorAdapter class."""

    @pytest.fixture
    def mock_standard_model(self):
        """Create a mock StandardLLMModel."""
        mock_model = Mock(spec=StandardLLMModel)
        mock_model.model_name = ModelName.OPENAI_GPT4
        mock_model.timeout_seconds = 60
        mock_model.get_model_name.return_value = ModelName.OPENAI_GPT4
        return mock_model

    @pytest.fixture
    def adapter(self, mock_standard_model):
        """Create StandardSectorAdapter with mock model."""
        return StandardSectorAdapter(mock_standard_model)

    def test_adapter_initialization(self, mock_standard_model):
        """Test adapter initialization."""
        adapter = StandardSectorAdapter(mock_standard_model)
        assert adapter.llm_model == mock_standard_model

    def test_get_model_name(self, adapter, mock_standard_model):
        """Test get_model_name method."""
        result = adapter.get_model_name()
        assert result == ModelName.OPENAI_GPT4
        mock_standard_model.get_model_name.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_sector_success(self, adapter, mock_standard_model):
        """Test successful sector analysis."""
        # Setup mock response
        mock_response = LLMResponse(
            content={
                "rating": 4,
                "summary": "Test summary",
                "sub_scores": {"growth": 4.0, "stability": 3.5, "valuation": 4.5},
                "weights": {"growth": 0.4, "stability": 0.3, "valuation": 0.3},
                "weighted_score": 4.0,
                "rationale": ["Strong fundamentals"],
                "references": [],
                "confidence": 0.85,
            },
            model=ModelName.OPENAI_GPT4,
            latency_ms=1500,
            timestamp_utc="2024-01-01T12:00:00Z",
            metadata={},
        )
        mock_standard_model.generate_structured = AsyncMock(return_value=mock_response)

        # Create request
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Mock validation functions
        with (
            patch(
                "sector_committee.scoring.llm_adapters.validate_sector_rating"
            ) as mock_validate,
            patch(
                "sector_committee.scoring.llm_adapters.create_openai_structured_output_schema"
            ) as mock_schema,
            patch(
                "sector_committee.scoring.llm_adapters.build_system_prompt"
            ) as mock_sys_prompt,
            patch(
                "sector_committee.scoring.llm_adapters.build_user_prompt"
            ) as mock_user_prompt,
        ):
            # Setup mocks
            mock_validate.return_value = Mock(is_valid=True, errors=[])
            mock_schema.return_value = {"type": "object"}
            mock_sys_prompt.return_value = "System prompt"
            mock_user_prompt.return_value = "User prompt"

            # Execute analysis
            result = await adapter.analyze_sector(request)

            # Verify result
            assert isinstance(result, ModelResult)
            assert result.model == ModelName.OPENAI_GPT4
            assert result.success is True
            assert result.latency_ms > 0
            assert result.data["rating"] == 4

            # Verify LLM model was called correctly
            mock_standard_model.generate_structured.assert_called_once()
            call_kwargs = mock_standard_model.generate_structured.call_args.kwargs
            assert call_kwargs["temperature"] == 0.1
            assert call_kwargs["max_tokens"] == 8000

    @pytest.mark.asyncio
    async def test_analyze_sector_validation_failure(
        self, adapter, mock_standard_model
    ):
        """Test sector analysis with validation failure."""
        # Setup mock response with invalid data
        mock_response = LLMResponse(
            content={"invalid": "data"},
            model=ModelName.OPENAI_GPT4,
            latency_ms=1500,
            timestamp_utc="2024-01-01T12:00:00Z",
            metadata={},
        )
        mock_standard_model.generate_structured = AsyncMock(return_value=mock_response)

        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Mock validation to fail
        with (
            patch(
                "sector_committee.scoring.llm_adapters.validate_sector_rating"
            ) as mock_validate,
            patch(
                "sector_committee.scoring.llm_adapters.create_openai_structured_output_schema"
            ),
            patch("sector_committee.scoring.llm_adapters.build_system_prompt"),
            patch("sector_committee.scoring.llm_adapters.build_user_prompt"),
        ):
            mock_validate.return_value = Mock(
                is_valid=False, errors=["Missing rating field"]
            )

            # Should raise SectorAnalysisError
            with pytest.raises(SectorAnalysisError) as exc_info:
                await adapter.analyze_sector(request)

            assert "Invalid response format" in str(exc_info.value)
            assert exc_info.value.model == ModelName.OPENAI_GPT4

    @pytest.mark.asyncio
    async def test_analyze_sector_llm_error(self, adapter, mock_standard_model):
        """Test sector analysis with LLM error."""
        # Setup LLM to raise error
        llm_error = LLMError(ModelName.OPENAI_GPT4, "API error", None)
        mock_standard_model.generate_structured = AsyncMock(side_effect=llm_error)

        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Mock other functions
        with (
            patch("sector_committee.scoring.llm_adapters.validate_sector_rating"),
            patch(
                "sector_committee.scoring.llm_adapters.create_openai_structured_output_schema"
            ),
            patch("sector_committee.scoring.llm_adapters.build_system_prompt"),
            patch("sector_committee.scoring.llm_adapters.build_user_prompt"),
        ):
            # Should raise SectorAnalysisError wrapping LLMError
            with pytest.raises(SectorAnalysisError) as exc_info:
                await adapter.analyze_sector(request)

            assert "LLM error: gpt-4: API error" in str(exc_info.value)
            assert exc_info.value.model == ModelName.OPENAI_GPT4
            assert exc_info.value.original_error == llm_error


class TestDeepResearchSectorAdapter:
    """Test DeepResearchSectorAdapter class."""

    @pytest.fixture
    def mock_two_stage_model(self):
        """Create a mock TwoStageLLMModel."""
        mock_model = Mock(spec=TwoStageLLMModel)
        mock_model.model_name = ModelName.OPENAI_O3_DEEP_RESEARCH
        mock_model.timeout_seconds = 300
        mock_model.get_model_name.return_value = ModelName.OPENAI_O3_DEEP_RESEARCH
        return mock_model

    @pytest.fixture
    def adapter(self, mock_two_stage_model):
        """Create DeepResearchSectorAdapter with mock model."""
        return DeepResearchSectorAdapter(mock_two_stage_model)

    def test_adapter_initialization(self, mock_two_stage_model):
        """Test adapter initialization."""
        adapter = DeepResearchSectorAdapter(mock_two_stage_model)
        assert adapter.llm_model == mock_two_stage_model

    def test_get_model_name(self, adapter, mock_two_stage_model):
        """Test get_model_name method."""
        result = adapter.get_model_name()
        assert result == ModelName.OPENAI_O3_DEEP_RESEARCH
        mock_two_stage_model.get_model_name.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_sector_success(self, adapter, mock_two_stage_model):
        """Test successful two-stage sector analysis."""
        # Setup mock responses
        research_response = LLMResponse(
            content="Comprehensive research content about the sector...",
            model=ModelName.OPENAI_O3_DEEP_RESEARCH,
            latency_ms=8000,
            timestamp_utc="2024-01-01T12:00:00Z",
            metadata={},
        )

        structured_response = LLMResponse(
            content={
                "rating": 4,
                "summary": "Test summary",
                "sub_scores": {"growth": 4.0, "stability": 3.5, "valuation": 4.5},
                "weights": {"growth": 0.4, "stability": 0.3, "valuation": 0.3},
                "weighted_score": 4.0,
                "rationale": ["Strong fundamentals"],
                "references": [
                    {
                        "url": "https://example.com",
                        "title": "Test",
                        "description": "Test ref",
                    }
                ],
                "confidence": 0.85,
            },
            model=ModelName.OPENAI_O3_DEEP_RESEARCH,
            latency_ms=2000,
            timestamp_utc="2024-01-01T12:01:00Z",
            metadata={},
        )

        mock_two_stage_model.generate_enhanced_reasoning = AsyncMock(
            return_value=research_response
        )
        mock_two_stage_model.generate_structured = AsyncMock(
            return_value=structured_response
        )

        # Create request
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Mock all the required functions
        with (
            patch(
                "sector_committee.scoring.llm_adapters.validate_sector_rating"
            ) as mock_validate,
            patch(
                "sector_committee.scoring.llm_adapters.create_openai_structured_output_schema"
            ) as mock_schema,
            patch(
                "sector_committee.scoring.llm_adapters.build_deep_research_system_prompt"
            ) as mock_research_sys_prompt,
            patch(
                "sector_committee.scoring.llm_adapters.build_deep_research_user_prompt"
            ) as mock_research_user_prompt,
            patch(
                "sector_committee.scoring.llm_adapters.build_structured_conversion_system_prompt"
            ) as mock_conv_sys,
            patch(
                "sector_committee.scoring.llm_adapters.build_conversion_prompt"
            ) as mock_conv_prompt,
            patch.object(
                adapter, "_process_references", new_callable=AsyncMock
            ) as mock_process_refs,
        ):
            # Setup mocks
            mock_validate.return_value = Mock(is_valid=True, errors=[])
            mock_schema.return_value = {"type": "object"}
            mock_research_sys_prompt.return_value = "Deep research system prompt"
            mock_research_user_prompt.return_value = "Deep research user prompt"
            mock_conv_sys.return_value = "Conversion system prompt"
            mock_conv_prompt.return_value = "Conversion prompt"

            # Execute analysis
            result = await adapter.analyze_sector(request)

            # Verify result
            assert isinstance(result, ModelResult)
            assert result.model == ModelName.OPENAI_O3_DEEP_RESEARCH
            assert result.success is True
            assert result.latency_ms > 0
            assert result.data["rating"] == 4

            # Verify both stages were called
            mock_two_stage_model.generate_enhanced_reasoning.assert_called_once()
            mock_two_stage_model.generate_structured.assert_called_once()

            # Verify reference processing was called
            mock_process_refs.assert_called_once()

    @pytest.mark.asyncio
    async def test_conduct_deep_research_failure(self, adapter, mock_two_stage_model):
        """Test deep research stage failure."""
        # Setup LLM to raise error in first stage
        llm_error = LLMError(ModelName.OPENAI_O3_DEEP_RESEARCH, "API error", None)
        mock_two_stage_model.generate_enhanced_reasoning = AsyncMock(
            side_effect=llm_error
        )

        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Mock required functions
        with (
            patch(
                "sector_committee.scoring.llm_adapters.build_deep_research_system_prompt"
            ),
            patch(
                "sector_committee.scoring.llm_adapters.build_deep_research_user_prompt"
            ),
        ):
            # Should raise SectorAnalysisError
            with pytest.raises(SectorAnalysisError) as exc_info:
                await adapter._conduct_deep_research(request)

            assert "Deep research stage failed" in str(exc_info.value)
            assert exc_info.value.model == ModelName.OPENAI_O3_DEEP_RESEARCH

    @pytest.mark.asyncio
    async def test_convert_to_structured_output_failure(
        self, adapter, mock_two_stage_model
    ):
        """Test structured output conversion failure."""
        # Setup LLM to raise error in second stage
        llm_error = LLMError(
            ModelName.OPENAI_O3_DEEP_RESEARCH, "Conversion error", None
        )
        mock_two_stage_model.generate_structured = AsyncMock(side_effect=llm_error)

        research_content = "Research content"
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        # Mock required functions
        with (
            patch(
                "sector_committee.scoring.llm_adapters.build_structured_conversion_system_prompt"
            ),
            patch("sector_committee.scoring.llm_adapters.build_conversion_prompt"),
            patch(
                "sector_committee.scoring.llm_adapters.create_openai_structured_output_schema"
            ),
        ):
            # Should raise SectorAnalysisError
            with pytest.raises(SectorAnalysisError) as exc_info:
                await adapter._convert_to_structured_output(research_content, request)

            assert "Structured output conversion failed" in str(exc_info.value)
            assert exc_info.value.model == ModelName.OPENAI_O3_DEEP_RESEARCH

    @pytest.mark.asyncio
    async def test_process_references(self, adapter):
        """Test reference processing."""
        request = SectorRequest(sector="Technology", horizon_weeks=4)
        references = [
            {"url": "https://example.com", "title": "Test Article"},
            {"url": "https://invalid-url", "title": "Invalid"},
        ]

        # Mock the download function
        async def mock_download(url):
            if url == "https://example.com":
                return True, "<html>Test content</html>"
            return False, None

        with (
            patch.object(adapter, "_download_url_content", side_effect=mock_download),
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            await adapter._process_references(references, request)

            # Verify references were updated
            assert references[0]["accessible"] is True
            assert "accessed_at" in references[0]
            assert "local_path" in references[0]

            assert references[1]["accessible"] is False
            assert "accessed_at" in references[1]
            assert "error" not in references[1]  # No error for inaccessible

    def test_sanitize_filename(self, adapter):
        """Test filename sanitization."""
        # Test invalid characters
        result = adapter._sanitize_filename('Test<>:"/\\|?*File')
        assert result == "Test_________File"

        # Test spaces
        result = adapter._sanitize_filename("Test   Multiple   Spaces")
        assert result == "Test_Multiple_Spaces"

        # Test length limit
        long_title = "A" * 100
        result = adapter._sanitize_filename(long_title)
        assert len(result) == 50

        # Test normal title
        result = adapter._sanitize_filename("Normal Title")
        assert result == "Normal_Title"


class TestSectorAdapterFactory:
    """Test SectorAdapterFactory class."""

    @pytest.mark.asyncio
    async def test_create_adapter_standard_model(self):
        """Test creating adapter for standard model."""
        with patch("sector_committee.scoring.llm_adapters.create_model") as mock_create:
            # Setup mock to return StandardLLMModel
            mock_model = Mock(spec=StandardLLMModel)
            mock_model.model_name = ModelName.OPENAI_GPT4
            mock_create.return_value = mock_model

            adapter = SectorAdapterFactory.create_adapter(ModelName.OPENAI_GPT4, 60)

            # Should create StandardSectorAdapter
            assert isinstance(adapter, StandardSectorAdapter)
            assert adapter.llm_model == mock_model
            mock_create.assert_called_once_with(ModelName.OPENAI_GPT4, 60)

    @pytest.mark.asyncio
    async def test_create_adapter_two_stage_model(self):
        """Test creating adapter for two-stage model."""
        with patch("sector_committee.scoring.llm_adapters.create_model") as mock_create:
            # Setup mock to return TwoStageLLMModel
            mock_model = Mock(spec=TwoStageLLMModel)
            mock_model.model_name = ModelName.OPENAI_O3_DEEP_RESEARCH
            mock_create.return_value = mock_model

            adapter = SectorAdapterFactory.create_adapter(
                ModelName.OPENAI_O3_DEEP_RESEARCH, 300
            )

            # Should create DeepResearchSectorAdapter
            assert isinstance(adapter, DeepResearchSectorAdapter)
            assert adapter.llm_model == mock_model
            mock_create.assert_called_once_with(ModelName.OPENAI_O3_DEEP_RESEARCH, 300)

    def test_create_adapter_unsupported_model_type(self):
        """Test creating adapter for unsupported model type."""
        with patch("sector_committee.scoring.llm_adapters.create_model") as mock_create:
            # Setup mock to return unknown model type
            mock_model = Mock(spec=LLMModel)  # Not Standard or TwoStage
            mock_model.model_name = ModelName.OPENAI_GPT4
            mock_create.return_value = mock_model

            # Should raise SectorAnalysisError
            with pytest.raises(SectorAnalysisError) as exc_info:
                SectorAdapterFactory.create_adapter(ModelName.OPENAI_GPT4, 60)

            assert "Unsupported LLM model type" in str(exc_info.value)

    def test_create_adapter_llm_error(self):
        """Test adapter creation with LLM error."""
        with patch("sector_committee.scoring.llm_adapters.create_model") as mock_create:
            # Setup create_model to raise LLMError
            llm_error = LLMError(ModelName.OPENAI_GPT4, "Creation failed", None)
            mock_create.side_effect = llm_error

            # Should raise SectorAnalysisError wrapping LLMError
            with pytest.raises(SectorAnalysisError) as exc_info:
                SectorAdapterFactory.create_adapter(ModelName.OPENAI_GPT4, 60)

            assert "Adapter creation failed" in str(exc_info.value)
            assert exc_info.value.original_error == llm_error

    @pytest.mark.asyncio
    async def test_create_default_adapter_standard(self):
        """Test creating default adapter with standard model."""
        with patch("sector_committee.llm_models.create_default_model") as mock_create:
            # Setup mock to return StandardLLMModel
            mock_model = Mock(spec=StandardLLMModel)
            mock_model.model_name = ModelName.OPENAI_GPT4O
            mock_create.return_value = mock_model

            adapter = SectorAdapterFactory.create_default_adapter(120)

            # Should create StandardSectorAdapter
            assert isinstance(adapter, StandardSectorAdapter)
            assert adapter.llm_model == mock_model
            mock_create.assert_called_once_with(120)

    @pytest.mark.asyncio
    async def test_create_default_adapter_two_stage(self):
        """Test creating default adapter with two-stage model."""
        with patch("sector_committee.llm_models.create_default_model") as mock_create:
            # Setup mock to return TwoStageLLMModel
            mock_model = Mock(spec=TwoStageLLMModel)
            mock_model.model_name = ModelName.OPENAI_O4_MINI_DEEP_RESEARCH
            mock_create.return_value = mock_model

            adapter = SectorAdapterFactory.create_default_adapter(300)

            # Should create DeepResearchSectorAdapter
            assert isinstance(adapter, DeepResearchSectorAdapter)
            assert adapter.llm_model == mock_model
            mock_create.assert_called_once_with(300)

    def test_create_default_adapter_llm_error(self):
        """Test default adapter creation with LLM error."""
        with patch("sector_committee.llm_models.create_default_model") as mock_create:
            # Setup create_default_model to raise LLMError
            llm_error = LLMError(ModelName.OPENAI_GPT4, "No default available", None)
            mock_create.side_effect = llm_error

            # Should raise SectorAnalysisError wrapping LLMError
            with pytest.raises(SectorAnalysisError) as exc_info:
                SectorAdapterFactory.create_default_adapter(300)

            assert "Default adapter creation failed" in str(exc_info.value)
            assert exc_info.value.original_error == llm_error


class TestSectorAnalysisAdapterInterface:
    """Test SectorAnalysisAdapter abstract interface."""

    def test_abstract_methods(self):
        """Test that SectorAnalysisAdapter is properly abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            SectorAnalysisAdapter(Mock())

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""

        class IncompleteAdapter(SectorAnalysisAdapter):
            pass  # Missing analyze_sector implementation

        mock_model = Mock(spec=LLMModel)

        # Should raise TypeError due to unimplemented abstract method
        with pytest.raises(TypeError):
            IncompleteAdapter(mock_model)

    def test_base_functionality(self):
        """Test base functionality works in concrete implementation."""

        class TestAdapter(SectorAnalysisAdapter):
            async def analyze_sector(self, request):
                return Mock()

        mock_model = Mock(spec=LLMModel)
        mock_model.get_model_name.return_value = ModelName.OPENAI_GPT4

        adapter = TestAdapter(mock_model)
        assert adapter.get_model_name() == ModelName.OPENAI_GPT4
        mock_model.get_model_name.assert_called_once()


class TestReferenceProcessing:
    """Test reference processing functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for reference processing tests."""
        mock_model = Mock(spec=TwoStageLLMModel)
        return DeepResearchSectorAdapter(mock_model)

    @pytest.mark.asyncio
    async def test_download_url_content_success(self, adapter):
        """Test successful URL content download."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Setup mock session and response
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<html>Test content</html>")

            # Setup the async context manager chain properly
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = Mock(return_value=mock_get_cm)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session_cm

            is_accessible, content = await adapter._download_url_content(
                "https://example.com"
            )

            assert is_accessible is True
            assert content == "<html>Test content</html>"

    @pytest.mark.asyncio
    async def test_download_url_content_failure(self, adapter):
        """Test URL content download failure."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Setup mock session and response with proper async context managers
            mock_session = AsyncMock()

            # Make get() raise an exception
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__ = AsyncMock(side_effect=Exception("Network error"))
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = Mock(return_value=mock_get_cm)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session_cm

            is_accessible, content = await adapter._download_url_content(
                "https://example.com"
            )

            assert is_accessible is False
            assert content is None

    @pytest.mark.asyncio
    async def test_download_url_content_404(self, adapter):
        """Test URL content download with 404 status."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            # Setup mock session and response with proper async context managers
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 404

            # Setup the async context manager chain properly
            mock_get_cm = AsyncMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session.get = Mock(return_value=mock_get_cm)

            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session_cm

            is_accessible, content = await adapter._download_url_content(
                "https://example.com"
            )

            assert is_accessible is False
            assert content is None
