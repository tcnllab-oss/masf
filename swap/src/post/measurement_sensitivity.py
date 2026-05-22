from swap.src.post.run_post_eval_engine import run_plan


def main(plan, force_rerun=False):
    run_plan(plan, force_rerun=force_rerun)