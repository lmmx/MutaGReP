from pathlib import Path

nfs_path = Path("/net/nfs.cirrascale/prior/zaidk")
mnms_path = nfs_path / "8-aluminium-chicken" / "src/codenav/eval_codebases/mnm"
m3eval_path = nfs_path / "8-aluminium-chicken" / "src/codenav/eval_codebases/m3eval"
phidata_path = nfs_path / "8-aluminium-chicken" / "src/codenav/eval_codebases/phidata"

longcode_arena_repos_root = (
    nfs_path
    / "JetBrains-Research__lca-library-based-code-generation/repos/mnt/data/shared-data/lca/repos_example_generation"
)
long_code_arena_repos = """1200wd__bitcoinlib
aidasoft__dd4hep
ansys__pyaedt
ansys__pydpf-core
ansys__pymapdl
avslab__basilisk
bokeh__bokeh
burnysc2__python-sc2
capytaine__capytaine
chalmersplasmatheory__dream
continualai__avalanche
deepmind__acme
dfki-ric__pytransform3d
dlr-rm__blenderproc
explosion__thinc
federatedai__fate
fortra__impacket
funkelab__gunpowder
fusion-power-plant-framework__bluemira
geodynamics__burnman
hewlettpackard__oneview-python
hiddensymmetries__simsopt
imsy-dkfz__simpa
iovisor__bcc
jchanvfx__nodegraphqt
kivy__kivy
kubernetes-client__python
labsn__expyfun
lightly-ai__lightly
manimcommunity__manim
matplotlib__basemap
microsoft__nni
microsoft__qlib
mne-tools__mne-python
nanophotonics__nplab
nucypher__nucypher
nvidia__nvflare
oarriaga__paz
paddlepaddle__fastdeploy
pmgbergen__porepy
pybamm-team__pybamm
pyomo__mpi-sppy
pyqtgraph__pyqtgraph
pyscf__pyscf
pysteps__pysteps
pytorch__torchrec
pyvista__pyvista
rlberry-py__rlberry
rstudio__py-shiny
scikit-learn__scikit-learn
seed-labs__seed-emulator
silnrsi__pysilfont
silx-kit__silx
simpeg__simpeg
smdogroup__tacs
stfc__psyclone
synerbi__sirf
unidata__metpy
urwid__urwid
vispy__vispy
weihuayi__fealpy
zulko__moviepy""".strip().splitlines()

known_repos = ["allenact"] + long_code_arena_repos + ["mnm", "m3eval"]


def get_es_index_name_for_repo(repo_name: str) -> str:
    return f"zk__{repo_name}"


def map_repo_to_path_on_filesystem(repo_name: str) -> Path:
    if repo_name == "mnm":
        return mnms_path
    if repo_name == "m3eval":
        return m3eval_path
    if repo_name == "phidata":
        return phidata_path
    if repo_name in long_code_arena_repos:
        return longcode_arena_repos_root / repo_name
    if repo_name == "allenact":
        return nfs_path / "8-aluminium-chicken/workspace" / "allenact"
    raise ValueError(f"Unknown repo: {repo_name}")
