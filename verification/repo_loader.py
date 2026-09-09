"""
Imports the *actual* network/loss code straight out of the jaxhrl repo, without
executing the `if __name__ == "__main__"` training scripts and without needing
the repo's broken/unrelated infra deps (brll_core, gymnax, mlflow, wandb,
jaxhrl.common.jax_wrappers -- which doesn't exist on disk at all).

We do this by pre-registering lightweight stand-ins for jaxhrl.common.{utils,
logger,wrappers,jax_wrappers} in sys.modules *before* importing the algorithm
files, so `from jaxhrl.common.utils import parse_config` etc. resolve to the
stand-in instead of touching the real (dependency-heavy) files on disk. None
of those symbols are used by the network/loss functions we import -- they're
only referenced inside each file's __main__ block.
"""
import importlib
import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path("/home/sam/Documents/jaxhrl")


def _install_stub(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def install_infra_stubs():
    _install_stub("jaxhrl.common.utils", parse_config=lambda: {})
    _install_stub("jaxhrl.common.logger", Logger=object)
    _install_stub(
        "jaxhrl.common.wrappers",
        make_jax_env=lambda *a, **k: None,
        run_eval_episode=lambda *a, **k: None,
    )
    _install_stub(
        "jaxhrl.common.jax_wrappers",
        make_jax_env=lambda *a, **k: None,
        run_eval_episode=lambda *a, **k: None,
    )
    # The newer algorithm files (option_critic.py, METRA.py) import their infra
    # from `brll_core.algorithms.common.*` instead of `jaxhrl.common.*`.
    for parent in ("brll_core", "brll_core.algorithms", "brll_core.algorithms.common"):
        if parent not in sys.modules:
            _install_stub(parent)
    _install_stub("brll_core.algorithms.common.utils", parse_config=lambda: {})
    _install_stub("brll_core.algorithms.common.logger", Logger=object)
    _install_stub(
        "brll_core.algorithms.common.jax_wrappers",
        make_jax_env=lambda *a, **k: None,
        run_eval_episode=lambda *a, **k: None,
    )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def load_dceo():
    install_infra_stubs()
    import jaxhrl.DCEO as dceo
    return dceo


def load_hdqn():
    """h-DQN.py has a hyphen in its filename, so it isn't a valid dotted
    import path -- load it directly from its file path instead."""
    install_infra_stubs()
    path = REPO_ROOT / "jaxhrl" / "h-DQN.py"
    spec = importlib.util.spec_from_file_location("jaxhrl_hdqn_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_option_keyboard():
    install_infra_stubs()
    import jaxhrl.option_keyboard as ok
    return ok


def load_hippo():
    install_infra_stubs()
    import jaxhrl.HiPPO as hippo
    return hippo


def load_hac():
    install_infra_stubs()
    import jaxhrl.HAC as hac
    return hac


def load_hierq():
    install_infra_stubs()
    import jaxhrl.HierQ as hierq
    return hierq


def load_option_critic():
    install_infra_stubs()
    import jaxhrl.option_critic as oc
    return oc


def load_moc():
    install_infra_stubs()
    import jaxhrl.MOC as moc
    return moc


def load_metra():
    install_infra_stubs()
    import jaxhrl.METRA as metra
    return metra


if __name__ == "__main__":
    dceo = load_dceo()
    print("DCEO OK:", dceo.LaplacianRepresentationNetwork, dceo.laplacian_loss_fn, dceo.q_loss_fn)
    hdqn = load_hdqn()
    print("h-DQN OK:", hdqn.QNetwork, hdqn.train_controller_step, hdqn.train_meta_step)
    hippo = load_hippo()
    print("HiPPO OK:", hippo.ManagerActorCritic, hippo.SkillActorCritic, hippo.select_hippo_action)
    hac = load_hac()
    print("HAC OK:", hac.Actor, hac.Critic, hac.train_level_step, hac.ring_add)
    hierq = load_hierq()
    print("HierQ OK:", hierq.update_level0, hierq.update_level_i, hierq.eps_greedy)
    oc = load_option_critic()
    print("Option-Critic OK:", oc.OptionCriticNetwork, oc.option_critic_loss_fn,
          oc.batch_select_option_critic_action)
    moc = load_moc()
    print("MOC OK:", moc.OptionCriticNetwork, moc.moc_loss_fn,
          moc.batch_select_option_critic_action)
    metra = load_metra()
    print("METRA OK:", metra.Encoder, metra.Actor, metra.metra_components,
          metra.sample_z, metra.metra_action)
