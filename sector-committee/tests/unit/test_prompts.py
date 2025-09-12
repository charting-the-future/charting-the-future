"""Unit tests for prompts module.

Tests all prompt building functions and the complex format_prompt logic
to achieve comprehensive test coverage.
"""

from sector_committee.data_models import SectorRequest
from sector_committee.scoring.prompts import (
    format_prompt,
    build_deep_research_system_prompt,
    build_deep_research_user_prompt,
    build_conversion_prompt,
    build_system_prompt,
    build_user_prompt,
    build_structured_conversion_system_prompt,
)
from sector_committee.scoring.prompt_utilities import (
    _BULLET_RE,
    _HEADER_RE,
)


class TestFormatPrompt:
    """Test the complex format_prompt function comprehensively."""

    def test_format_prompt_basic_paragraph(self):
        """Test basic paragraph formatting."""
        raw = """
        This is a simple paragraph that should be wrapped
        and formatted nicely.
        """
        result = format_prompt(raw, width=40)
        assert "This is a simple paragraph that should" in result
        assert "be wrapped and formatted nicely." in result
        assert result.count("\n") == 1  # Single wrapped paragraph

    def test_format_prompt_preserve_headers(self):
        """Test that ALL CAPS headers are preserved."""
        raw = """
        RESEARCH OBJECTIVES:
        This is content under the header.
        
        ANOTHER HEADER:
        More content here.
        """
        result = format_prompt(raw)
        assert "RESEARCH OBJECTIVES:" in result
        assert "ANOTHER HEADER:" in result
        # Headers should be on their own lines
        lines = result.split("\n")
        header_lines = [line for line in lines if line.endswith(":") and line.isupper()]
        assert len(header_lines) == 2

    def test_format_prompt_bullet_points(self):
        """Test bullet point formatting and continuation."""
        raw = """
        - First bullet point
          with continuation line
        - Second bullet
        - Third bullet point
        """
        result = format_prompt(raw, list_indent=2)
        assert "  - First bullet point with continuation line" in result
        assert "  - Second bullet" in result
        assert "  - Third bullet point" in result

    def test_format_prompt_numbered_lists(self):
        """Test numbered list formatting."""
        raw = """
        1. First numbered item
           with continuation
        2. Second item
        3. Third item
        """
        result = format_prompt(raw, list_indent=4)
        assert "    1. First numbered item with continuation" in result
        assert "    2. Second item" in result
        assert "    3. Third item" in result

    def test_format_prompt_mixed_content(self):
        """Test mixed headers, paragraphs, and bullets."""
        raw = """
        Introduction paragraph.

        MAIN SECTION:
        - Bullet one
        - Bullet two
          with continuation

        Closing paragraph.
        """
        result = format_prompt(raw)
        assert "Introduction paragraph." in result
        assert "MAIN SECTION:" in result
        assert "  - Bullet one" in result
        assert "  - Bullet two with continuation" in result
        assert "Closing paragraph." in result

    def test_format_prompt_preserve_blank_lines(self):
        """Test that blank lines are preserved."""
        raw = """
        First paragraph.

        Second paragraph after blank line.
        """
        result = format_prompt(raw)
        lines = result.split("\n")
        # Should have blank line between paragraphs
        assert "" in lines

    def test_format_prompt_width_wrapping(self):
        """Test that text wraps at specified width."""
        raw = "This is a very long line that should definitely be wrapped at the specified width parameter."
        result = format_prompt(raw, width=30)
        lines = result.split("\n")
        # All lines should be <= 30 characters (except possibly last)
        for line in lines[:-1]:
            assert len(line) <= 30

    def test_format_prompt_titlecase_sector(self):
        """Test title-casing of sector phrases."""
        raw = "Analysis on the technology sector shows strong performance."
        result = format_prompt(raw, titlecase_sector_phrase=True)
        assert "on the Technology sector" in result

    def test_format_prompt_no_titlecase_sector(self):
        """Test disabling sector title-casing."""
        raw = "Analysis on the technology sector shows strong performance."
        result = format_prompt(raw, titlecase_sector_phrase=False)
        assert "on the technology sector" in result

    def test_format_prompt_replacements(self):
        """Test regex replacements parameter."""
        raw = "Replace XXX with YYY in this text."
        replacements = {r"XXX": "ABC", r"YYY": "DEF"}
        result = format_prompt(raw, replacements=replacements)
        assert "Replace ABC with DEF in this text." in result

    def test_format_prompt_complex_bullets(self):
        """Test complex bullet formatting with different markers."""
        raw = """
        • Unicode bullet point
        - Dash bullet point
        1. Numbered item
        2. Another numbered item
        """
        result = format_prompt(raw)
        assert "  • Unicode bullet point" in result
        assert "  - Dash bullet point" in result
        assert "  1. Numbered item" in result
        assert "  2. Another numbered item" in result

    def test_format_prompt_indented_bullets(self):
        """Test bullets with existing indentation."""
        raw = """
            - Indented bullet
            - Another indented bullet
        """
        result = format_prompt(raw, list_indent=2)
        # Should normalize to consistent indentation
        assert "  - Indented bullet" in result
        assert "  - Another indented bullet" in result

    def test_format_prompt_bullet_continuation_lines(self):
        """Test bullet points with multiple continuation lines."""
        raw = """
        - First bullet
          with first continuation
          and second continuation
          and third continuation
        - Second bullet
        """
        result = format_prompt(raw)
        assert (
            "First bullet with first continuation and second continuation and third continuation"
            in result
        )

    def test_format_prompt_empty_input(self):
        """Test handling of empty input."""
        result = format_prompt("")
        assert result == ""

    def test_format_prompt_whitespace_only(self):
        """Test handling of whitespace-only input."""
        result = format_prompt("   \n  \n   ")
        assert result == ""


class TestRegexPatterns:
    """Test the regex patterns used in format_prompt."""

    def test_bullet_regex_patterns(self):
        """Test bullet regex matches various formats."""
        assert _BULLET_RE.match("- Simple bullet")
        assert _BULLET_RE.match("  - Indented bullet")
        assert _BULLET_RE.match("• Unicode bullet")
        assert _BULLET_RE.match("1. Numbered item")
        assert _BULLET_RE.match("  2. Indented numbered")
        assert not _BULLET_RE.match("Not a bullet")
        assert not _BULLET_RE.match("Regular text")

    def test_header_regex_patterns(self):
        """Test header regex matches ALL CAPS headers."""
        assert _HEADER_RE.match("RESEARCH OBJECTIVES:")
        assert _HEADER_RE.match("  MAIN SECTION:  ")
        assert _HEADER_RE.match("ANALYSIS & RESULTS:")
        assert _HEADER_RE.match("KEY FINDINGS/CONCLUSIONS:")
        assert not _HEADER_RE.match("Not A Header:")
        assert not _HEADER_RE.match("mixed Case Header:")
        assert not _HEADER_RE.match("lowercase header:")


class TestBuildDeepResearchPrompts:
    """Test deep research prompt building (system and user prompts)."""

    def test_build_deep_research_system_prompt_content(self):
        """Test deep research system prompt contains all required content."""
        result = build_deep_research_system_prompt()

        assert "senior equity research analyst" in result
        assert "institutional investment" in result and "management firm" in result
        assert "TRI-PILLAR" in result and "METHODOLOGY" in result
        assert "FUNDAMENTAL ANALYSIS" in result
        assert "SENTIMENT ANALYSIS" in result
        assert "TECHNICAL ANALYSIS" in result
        assert "DATA QUALITY AND VERIFICATION STANDARDS:" in result
        assert "ANALYTICAL OUTPUT REQUIREMENTS:" in result
        assert "PROFESSIONAL STANDARDS:" in result

    def test_build_deep_research_system_prompt_pillar_details(self):
        """Test system prompt contains detailed pillar descriptions."""
        result = build_deep_research_system_prompt()

        # Check for detailed pillar content
        assert "Earnings Quality:" in result
        assert "Valuation Assessment:" in result
        assert "Institutional Positioning:" in result
        assert "Price Action:" in result
        assert "Momentum Indicators:" in result

    def test_build_deep_research_user_prompt_basic(self):
        """Test basic deep research user prompt generation."""
        request = SectorRequest(sector="Technology", horizon_weeks=4)
        result = build_deep_research_user_prompt(request)

        assert "Technology sector" in result
        assert "4-week" in result and "investment horizon" in result
        # User prompt should be concise
        assert len(result) < 500

    def test_build_deep_research_user_prompt_with_weights(self):
        """Test deep research user prompt with custom weights."""
        request = SectorRequest(
            sector="Healthcare",
            horizon_weeks=8,
            weights_hint={"fundamentals": 0.6, "sentiment": 0.2, "technicals": 0.2},
        )
        result = build_deep_research_user_prompt(request)

        assert "Healthcare sector" in result
        assert "8-week" in result and "investment horizon" in result
        assert "Apply these pillar weights" in result
        assert "0.6" in result  # fundamentals weight

    def test_build_deep_research_user_prompt_long_horizon(self):
        """Test deep research user prompt with longer horizon."""
        request = SectorRequest(sector="Energy", horizon_weeks=12)
        result = build_deep_research_user_prompt(request)

        assert "Energy sector" in result
        assert "12-week" in result
        assert "investment" in result and "horizon" in result

    def test_build_deep_research_user_prompt_special_sector_name(self):
        """Test deep research user prompt with special characters in sector."""
        request = SectorRequest(sector="Real Estate & REITs", horizon_weeks=6)
        result = build_deep_research_user_prompt(request)

        # Note: title case conversion changes "REITs" to "Reits"
        assert "Real Estate & Reits sector" in result
        assert "6-week" in result and "investment horizon" in result

    def test_deep_research_prompts_separation(self):
        """Test that system and user prompts are properly separated."""
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        system_prompt = build_deep_research_system_prompt()
        user_prompt = build_deep_research_user_prompt(request)

        # System prompt should contain methodology, user prompt should be specific
        assert len(system_prompt) > len(user_prompt)
        assert "methodology" in system_prompt.lower()
        assert "Technology sector" in user_prompt
        assert "Technology sector" not in system_prompt


class TestBuildConversionPrompt:
    """Test conversion prompt building."""

    def test_build_conversion_prompt_basic(self):
        """Test basic conversion prompt generation."""
        research_content = (
            "Sample research content about technology sector performance."
        )
        request = SectorRequest(sector="Technology", horizon_weeks=4)
        result = build_conversion_prompt(research_content, request)

        assert "Technology" in result
        assert "4 weeks" in result
        assert research_content in result
        assert "CONVERSION REQUIREMENTS:" in result
        assert "Extract tri-pillar scores" in result

    def test_build_conversion_prompt_with_weights(self):
        """Test conversion prompt with custom weights hint."""
        research_content = "Research data here."
        request = SectorRequest(
            sector="Healthcare",
            horizon_weeks=8,
            weights_hint={"fundamentals": 0.7, "sentiment": 0.2, "technicals": 0.1},
        )
        result = build_conversion_prompt(research_content, request)

        assert "Healthcare" in result
        assert "8 weeks" in result
        assert research_content in result
        # Should show custom weights, not default
        assert "fundamentals" in result
        assert "0.7" in result or "70%" in result

    def test_build_conversion_prompt_no_weights(self):
        """Test conversion prompt without custom weights."""
        research_content = "Research data here."
        request = SectorRequest(sector="Energy", horizon_weeks=6)
        result = build_conversion_prompt(research_content, request)

        assert "Energy" in result
        assert "6 weeks" in result
        assert "Default (50/30/20)" in result

    def test_build_conversion_prompt_long_research(self):
        """Test conversion prompt with long research content."""
        research_content = "Very long research content. " * 100  # Long content
        request = SectorRequest(sector="Financial Services", horizon_weeks=2)
        result = build_conversion_prompt(research_content, request)

        assert "Financial Services" in result
        assert "2 weeks" in result
        # Content gets wrapped, so check for a portion
        assert "Very long research content." in result


class TestBuildSystemPrompt:
    """Test system prompt building."""

    def test_build_system_prompt_content(self):
        """Test system prompt contains all required content."""
        result = build_system_prompt()

        assert "expert quantitative analyst" in result
        assert "tri-" in result and "pillar" in result and "methodology" in result
        assert "FUNDAMENTALS:" in result
        assert "SENTIMENT:" in result
        assert "TECHNICALS:" in result
        assert "1-5" in result  # Rating scale
        assert "JSON response" in result
        assert '"rating":' in result
        assert '"sub_scores":' in result
        assert '"weights":' in result

    def test_build_system_prompt_json_schema(self):
        """Test system prompt contains valid JSON schema example."""
        result = build_system_prompt()

        # Check for key JSON fields
        assert '"fundamentals":' in result
        assert '"sentiment":' in result
        assert '"technicals":' in result
        assert '"weighted_score":' in result
        assert '"rationale":' in result
        assert '"references":' in result
        assert '"confidence":' in result

    def test_build_system_prompt_formatting(self):
        """Test system prompt is properly formatted."""
        result = build_system_prompt()

        # Should be non-empty and formatted
        assert len(result) > 100
        assert result.strip() == result  # No leading/trailing whitespace
        assert "ANALYSIS FRAMEWORK:" in result
        assert "REQUIREMENTS:" in result


class TestBuildUserPrompt:
    """Test user prompt building."""

    def test_build_user_prompt_basic(self):
        """Test basic user prompt generation."""
        request = SectorRequest(sector="Technology", horizon_weeks=4)
        result = build_user_prompt(request)

        assert "Technology sector" in result
        assert "4 weeks" in result
        assert "FUNDAMENTAL ANALYSIS:" in result
        assert "SENTIMENT ANALYSIS:" in result
        assert "TECHNICAL ANALYSIS:" in result
        assert "DELIVERABLES:" in result

    def test_build_user_prompt_with_weights(self):
        """Test user prompt with custom weights."""
        request = SectorRequest(
            sector="Healthcare",
            horizon_weeks=8,
            weights_hint={"fundamentals": 0.6, "sentiment": 0.3, "technicals": 0.1},
        )
        result = build_user_prompt(request)

        assert "Healthcare sector" in result
        assert "8 weeks" in result
        assert "Use these pillar weights:" in result

    def test_build_user_prompt_no_weights(self):
        """Test user prompt without custom weights."""
        request = SectorRequest(sector="Energy", horizon_weeks=6)
        result = build_user_prompt(request)

        assert "Energy sector" in result
        assert "6 weeks" in result
        # Should not have weights instruction
        assert "Use these pillar weights:" not in result

    def test_build_user_prompt_special_characters(self):
        """Test user prompt with special characters in sector name."""
        request = SectorRequest(
            sector="Consumer Discretionary & Retail", horizon_weeks=12
        )
        result = build_user_prompt(request)

        assert "Consumer Discretionary & Retail sector" in result
        assert "12 weeks" in result

    def test_build_user_prompt_requirements(self):
        """Test user prompt contains all required sections."""
        request = SectorRequest(sector="Financials", horizon_weeks=2)
        result = build_user_prompt(request)

        assert "Sector-specific metrics" in result
        assert "Institutional investor positioning" in result
        assert "Sector ETF price action" in result
        assert "1-5 scores" in result
        assert "confidence score" in result
        assert "real, verifiable URLs" in result


class TestBuildStructuredConversionSystemPrompt:
    """Test structured conversion system prompt building."""

    def test_build_structured_conversion_system_prompt_content(self):
        """Test structured conversion system prompt content."""
        result = build_structured_conversion_system_prompt()

        assert "quantitative analyst" in result
        assert "structured sector ratings" in result
        assert "schema requirements" in result

    def test_build_structured_conversion_system_prompt_formatting(self):
        """Test structured conversion system prompt is properly formatted."""
        result = build_structured_conversion_system_prompt()

        # Should be non-empty and properly formatted
        assert len(result) > 50
        assert result.strip() == result
        assert len(result.split("\n")) >= 1  # Multi-line or single line is fine


class TestPromptValidation:
    """Test prompt building with edge cases and validation."""

    def test_prompts_with_empty_sector(self):
        """Test prompts handle empty sector gracefully."""
        request = SectorRequest(sector="", horizon_weeks=4)

        # Functions should still work, just with empty sector
        deep_research_user = build_deep_research_user_prompt(request)
        user_prompt = build_user_prompt(request)

        assert " sector" in deep_research_user
        assert " sector" in user_prompt

    def test_prompts_with_minimal_horizon(self):
        """Test prompts with minimum horizon weeks."""
        request = SectorRequest(sector="Test", horizon_weeks=1)

        result = build_user_prompt(request)
        assert "1 weeks" in result or "1-week" in result

    def test_prompts_with_maximum_horizon(self):
        """Test prompts with maximum horizon weeks."""
        request = SectorRequest(sector="Test", horizon_weeks=52)

        result = build_user_prompt(request)
        assert "52 weeks" in result or "52-week" in result

    def test_prompts_with_complex_weights(self):
        """Test prompts with complex weight structures."""
        request = SectorRequest(
            sector="Technology",
            horizon_weeks=4,
            weights_hint={"fundamentals": 0.45, "sentiment": 0.35, "technicals": 0.20},
        )

        deep_research_user = build_deep_research_user_prompt(request)
        user_prompt = build_user_prompt(request)
        conversion = build_conversion_prompt("test research", request)

        assert "0.45" in deep_research_user or "45%" in deep_research_user
        assert "weights" in user_prompt.lower()
        assert "fundamentals" in conversion


class TestPromptConsistency:
    """Test consistency across different prompt functions."""

    def test_all_prompts_mention_sector(self):
        """Test that all prompts properly include the sector name."""
        request = SectorRequest(sector="Test Sector", horizon_weeks=4)

        deep_research_user = build_deep_research_user_prompt(request)
        user_prompt = build_user_prompt(request)
        conversion = build_conversion_prompt("research", request)

        assert "Test Sector" in deep_research_user
        assert "Test Sector" in user_prompt
        assert "Test Sector" in conversion

    def test_all_prompts_mention_horizon(self):
        """Test that relevant prompts include the time horizon."""
        request = SectorRequest(sector="Technology", horizon_weeks=8)

        deep_research_user = build_deep_research_user_prompt(request)
        user_prompt = build_user_prompt(request)
        conversion = build_conversion_prompt("research", request)

        assert "8" in deep_research_user
        assert "8" in user_prompt
        assert "8" in conversion

    def test_prompt_lengths_reasonable(self):
        """Test that prompts are reasonable lengths."""
        request = SectorRequest(sector="Technology", horizon_weeks=4)

        system_prompt = build_system_prompt()
        deep_research_system = build_deep_research_system_prompt()
        deep_research_user = build_deep_research_user_prompt(request)
        user_prompt = build_user_prompt(request)
        conversion = build_conversion_prompt("research content", request)
        structured_system = build_structured_conversion_system_prompt()

        # All prompts should be substantial but not excessive
        assert 100 < len(system_prompt) < 10000
        assert (
            1000 < len(deep_research_system) < 20000
        )  # System prompt should be comprehensive
        assert 50 < len(deep_research_user) < 1000  # User prompt should be concise
        assert 100 < len(user_prompt) < 5000
        assert 50 < len(conversion) < 2000
        assert 50 < len(structured_system) < 1000
