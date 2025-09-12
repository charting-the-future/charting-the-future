#!/usr/bin/env python3
"""
Demonstration script for Deep Research API integration.

This script demonstrates the complete o4-mini-deep-research integration including:
- Single model approach (no ensemble)
- Robust JSON parsing from text responses
- Reference downloading and archiving
- Comprehensive audit logging
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from sector_committee.scoring.factory import ModelFactory
from sector_committee.data_models import SectorRequest
from sector_committee.llm_models import ModelName
from sector_committee.scoring.audit import AuditLogger


async def demo_deep_research_integration():
    """Demonstrate complete Deep Research API integration."""
    print("🚀 Deep Research API Integration Demo")
    print("=" * 50)

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set.")
        print("   Set your OpenAI API key to run live API tests:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()
        print("📋 Running integration tests without live API calls...")
        await demo_offline_functionality()
        return

    print("✅ OpenAI API key configured")
    print("🔬 Running live Deep Research API integration...")
    print()

    # Initialize components
    client = ModelFactory.create_default_client(
        timeout_seconds=120
    )  # 2 minute timeout for demo
    audit_logger = AuditLogger()

    # Create analysis ID
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    analysis_id = f"{timestamp}_DEMO_DEEP_RESEARCH"

    print(f"📊 Analysis ID: {analysis_id}")

    # Create test request
    request = SectorRequest(
        sector="Information Technology",
        horizon_weeks=4,
        weights_hint={"fundamentals": 0.4, "sentiment": 0.4, "technicals": 0.2},
    )

    print(f"🎯 Analyzing sector: {request.sector}")
    print(f"📅 Horizon: {request.horizon_weeks} weeks")
    print()

    try:
        # Log analysis start
        await audit_logger.log_analysis_start(analysis_id, request)
        print("📝 Analysis start logged")

        # Perform analysis
        print("🤖 Calling o4-mini-deep-research model...")
        print("   This may take 1-2 minutes due to web search...")

        result = await client.analyze_sector(request)

        print("✅ Analysis completed!")
        print(f"   Latency: {result.latency_ms:.0f}ms")
        print(f"   Cost: ${result.cost_usd:.4f}")
        print(f"   Rating: {result.data['rating']}/5")
        print(f"   Confidence: {result.data['confidence']:.2f}")

        # Check references
        references = result.data.get("references", [])
        print(f"📚 References found: {len(references)}")

        if references:
            accessible_count = sum(
                1 for ref in references if ref.get("accessible", False)
            )
            downloaded_count = sum(1 for ref in references if ref.get("local_path"))

            print(f"   Accessible: {accessible_count}/{len(references)}")
            print(f"   Downloaded: {downloaded_count}/{len(references)}")

            # Show sample references
            print("   Sample references:")
            for i, ref in enumerate(references[:3]):
                status = "✅" if ref.get("accessible") else "❌"
                print(f"     {i + 1}. {status} {ref.get('title', 'Unknown title')}")
                print(f"        {ref.get('url', 'No URL')}")

        # Log completion
        await audit_logger.log_analysis_completion(analysis_id, result.data, result)
        print("📝 Analysis completion logged")

        # Show audit trail
        print()
        print("📋 Audit Trail:")

        # Check for generated files
        logs_dir = Path("logs/audit")
        if logs_dir.exists():
            analysis_files = list(logs_dir.rglob(f"*{analysis_id}*"))
            print(f"   Generated {len(analysis_files)} audit files:")
            for file_path in analysis_files:
                print(f"     - {file_path}")

        # Check for downloaded references
        ref_dir = Path("logs/audit/references")
        if ref_dir.exists():
            ref_dirs = list(
                ref_dir.glob(f"*{request.sector.replace(' ', '_').upper()}*")
            )
            if ref_dirs:
                ref_files = list(ref_dirs[0].glob("*.html")) if ref_dirs else []
                print(f"   Downloaded {len(ref_files)} reference files to:")
                for ref_dir_path in ref_dirs:
                    print(f"     - {ref_dir_path}")

        print()
        print("🎉 Demo completed successfully!")
        print("🔍 Key improvements over previous implementation:")
        print("   • Uses o4-mini-deep-research with real web search")
        print("   • No more hallucinated URLs")
        print("   • Robust JSON parsing from text responses")
        print("   • Automatic reference downloading and archiving")
        print("   • Enhanced audit logging with reference tracking")

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        await audit_logger.log_analysis_failure(analysis_id, request.sector, str(e))
        raise


async def demo_offline_functionality():
    """Demonstrate functionality without live API calls."""
    print("🔧 Testing offline functionality...")
    print()

    # Test model factory
    print("1. Testing ModelFactory:")
    try:
        client = ModelFactory.create_default_client()
        supported_models = ModelFactory.get_supported_models()
        print(f"   ✅ Default client: {client.get_model_name().value}")
        print(f"   ✅ Supported models: {[m.value for m in supported_models]}")
    except Exception as e:
        print(f"   ❌ Factory test failed: {e}")

    # Test JSON parsing
    print("\n2. Testing JSON parsing:")
    try:
        test_client = ModelFactory.create(ModelName.OPENAI_O4_MINI_DEEP_RESEARCH)

        # Test various JSON formats
        test_cases = [
            ("Direct JSON", '{"rating": 4, "confidence": 0.8}'),
            ("Markdown JSON", '```json\n{"rating": 3, "confidence": 0.7}\n```'),
            ("Embedded JSON", 'Analysis: {"rating": 5, "confidence": 0.9} Complete.'),
        ]

        for name, content in test_cases:
            try:
                result = await test_client._parse_json_response(content)
                print(f"   ✅ {name}: {result}")
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")

    except Exception as e:
        print(f"   ❌ JSON parsing test failed: {e}")

    # Test audit logger
    print("\n3. Testing audit logger:")
    try:
        audit_logger = AuditLogger()
        print("   ✅ Audit logger initialized")
        print(f"   ✅ Log directory: {audit_logger.log_directory}")

        # Test log directory creation
        test_dirs = [
            audit_logger.analysis_logs_dir,
            audit_logger.error_logs_dir,
            audit_logger.performance_logs_dir,
        ]

        for log_dir in test_dirs:
            if log_dir.exists():
                print(f"   ✅ Directory exists: {log_dir.name}")
            else:
                print(f"   ❌ Directory missing: {log_dir.name}")

    except Exception as e:
        print(f"   ❌ Audit logger test failed: {e}")

    print("\n✅ Offline functionality tests completed!")
    print("\n💡 To test with live API calls:")
    print("   1. Set OPENAI_API_KEY environment variable")
    print("   2. Run this script again")
    print("   3. The system will make real API calls to o4-mini-deep-research")


if __name__ == "__main__":
    asyncio.run(demo_deep_research_integration())
