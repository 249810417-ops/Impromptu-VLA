try:
    from .datasets import *
    from .models import *
    from .apis import *
    from .core.evaluation import *
except Exception:
    # Allow lightweight imports (for example PlanningMetric or cached-info paths)
    # in environments without the full OpenMMLab stack.
    pass
