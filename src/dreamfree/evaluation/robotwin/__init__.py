"""Policy callbacks consumed by the RoboTwin evaluation engine."""


def get_model(options):
    from dreamfree.evaluation.policy import build_policy

    return build_policy("robotwin", options)


def eval(task_env, model, observation):
    model.step(task_env, observation)


def reset_model(model):
    model.reset()
