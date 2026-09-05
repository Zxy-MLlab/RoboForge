from __future__ import annotations

from pathlib import Path


class ProjectWorkspace:
    """A normal editable project tree owned by an OpenHands conversation."""

    REQUIRED_DIRS = (
        "controllers", "robot_sdk", "capabilities/perception",
        "capabilities/geometry", "capabilities/grasping",
        "capabilities/planning", "capabilities/ik", "capabilities/control",
        "runtime_adapters", "services", "models", "tests", "configs",
        "requirements",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def initialize(self, *, entrypoint: str = "controllers/controller.py") -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in self.REQUIRED_DIRS:
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        path = self.root / entrypoint
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                '"""Initial intentionally incomplete candidate."""\n\n'
                "def run(robot):\n    return robot.observe(channel='proprioception', request={})\n",
                encoding="utf-8",
            )
        # Bootstrap the editable Robot Stack into the same workspace as the
        # Controller.  These are ordinary source files: OpenHands may inspect,
        # test, replace, or remove them before the next bundle freeze.
        package_root = Path(__file__).resolve().parents[2]
        bootstrap = {
            "robot_sdk/franka_libero_api.py": package_root / "embodied_codex/adapters/franka_libero_api.py",
            "robot_sdk/libero_sdk.py": package_root / "embodied_codex/adapters/libero_sdk.py",
            "runtime_adapters/libero.py": package_root / "embodied_codex/adapters/libero.py",
            "runtime_adapters/franka_deployment.py": package_root / "embodied_codex/deployments/libero.py",
        }
        for relative, source in bootstrap.items():
            destination = self.root / relative
            if not destination.exists() and source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
        capability_root = package_root / "embodied_codex/capabilities"
        for source in sorted(capability_root.glob("*.py")):
            destination = self.root / "capabilities" / source.name
            if not destination.exists():
                destination.write_bytes(source.read_bytes())
        manifest = self.root / "robot_sdk/BOOTSTRAP_SOURCES.json"
        if not manifest.exists():
            manifest.write_text(
                '{"aspire": "f4c8939aab0af9b97690c561bd80e282940f7886", '
                '"cap_x": "53e9966d7a8e2fa7494676772bccc35280f5c0ed", '
                '"license": "Apache-2.0/MIT and MIT", '
                '"note": "bootstrap copies are editable workspace code"}\n',
                encoding="utf-8",
            )
        return path

    def inside(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False
