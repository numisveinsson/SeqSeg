"""SeqSeg public API (lazy imports for heavy dependencies)."""

__all__ = (
    "AlgorithmConfig",
    "NnUNetModelSpec",
    "TracingLimits",
    "CaseIO",
    "trace_centerline",
    "trace_centerline_from_context",
    "TracingContext",
    "TracingResult",
    "load_yaml_config",
    "run_classic_batch",
    "run_plus_batch",
    "bootstrap_simvascular_project",
    "bootstrap_simvascular_project_batch",
    "run_global_centerline_single",
    "run_global_centerline_batch",
    "BranchSeed",
    "branch_seed_at_point",
    "seeds_to_potential_branches",
    "TracingOptions",
    "run_tracing",
)


def __getattr__(name: str):
    if name == "AlgorithmConfig":
        from seqseg.config_models import AlgorithmConfig

        return AlgorithmConfig
    if name == "NnUNetModelSpec":
        from seqseg.config_models import NnUNetModelSpec

        return NnUNetModelSpec
    if name == "TracingLimits":
        from seqseg.config_models import TracingLimits

        return TracingLimits
    if name == "CaseIO":
        from seqseg.config_models import CaseIO

        return CaseIO
    if name == "trace_centerline":
        from seqseg.modules.tracing import trace_centerline

        return trace_centerline
    if name == "trace_centerline_from_context":
        from seqseg.modules.tracing import trace_centerline_from_context

        return trace_centerline_from_context
    if name == "TracingContext":
        from seqseg.modules.tracing import TracingContext

        return TracingContext
    if name == "TracingResult":
        from seqseg.modules.tracing import TracingResult

        return TracingResult
    if name == "load_yaml_config":
        from seqseg.config_models import load_yaml_config

        return load_yaml_config
    if name == "run_classic_batch":
        from seqseg.pipeline.classic import run_classic_batch

        return run_classic_batch
    if name == "run_plus_batch":
        from seqseg.pipeline.plus import run_plus_batch

        return run_plus_batch
    if name == "bootstrap_simvascular_project":
        from seqseg.pipeline.post import bootstrap_simvascular_project

        return bootstrap_simvascular_project
    if name == "bootstrap_simvascular_project_batch":
        from seqseg.pipeline.post import bootstrap_simvascular_project_batch

        return bootstrap_simvascular_project_batch
    if name == "run_global_centerline_single":
        from seqseg.pipeline.post import run_global_centerline_single

        return run_global_centerline_single
    if name == "run_global_centerline_batch":
        from seqseg.pipeline.post import run_global_centerline_batch

        return run_global_centerline_batch
    if name == "BranchSeed":
        from seqseg.api import BranchSeed

        return BranchSeed
    if name == "branch_seed_at_point":
        from seqseg.api import branch_seed_at_point

        return branch_seed_at_point
    if name == "seeds_to_potential_branches":
        from seqseg.api import seeds_to_potential_branches

        return seeds_to_potential_branches
    if name == "TracingOptions":
        from seqseg.api import TracingOptions

        return TracingOptions
    if name == "run_tracing":
        from seqseg.api import run_tracing

        return run_tracing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
