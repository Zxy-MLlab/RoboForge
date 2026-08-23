class HarnessError(RuntimeError):
    pass


class NodeCompileError(HarnessError):
    pass


class GraphCompileError(HarnessError):
    pass


class NodeRuntimeError(HarnessError):
    pass


class AssetError(HarnessError):
    pass
