"""Load and view a partial MuJoCo model."""

import os
import mujoco
import mujoco.viewer


def get_chain(xml_path: str, link_name: str) -> list[str]:
    """Get kinematic chain from root to link."""
    spec = mujoco.MjSpec.from_file(xml_path)
    body = spec.worldbody.find_child(link_name)

    chain = []
    while body and body.name != 'world':
        chain.append(body.name)
        body = body.parent
    return chain[::-1]


def load_partial(xml_path: str, keep: list[str]) -> mujoco.MjModel:
    """Load model keeping only specified bodies."""
    spec = mujoco.MjSpec.from_file(xml_path)
    keep_set = set(keep) | {'world', spec.worldbody.first_body().name}

    to_delete = [b.name for b in spec.bodies if b.name not in keep_set]
    for name in reversed(to_delete):
        body = spec.worldbody.find_child(name)
        if body:
            spec.delete(body)
    return spec.compile()


def print_tree(xml_path: str):
    """Print body tree (chains collapsed)."""
    spec = mujoco.MjSpec.from_file(xml_path)

    def get_children(body):
        children = []
        child = body.first_body()
        while child:
            children.append(child)
            child = body.next_body(child)
        return children

    def print_node(body, indent="", last=True):
        # Collapse linear chains
        chain = [body.name]
        children = get_children(body)
        while len(children) == 1:
            body = children[0]
            chain.append(body.name)
            children = get_children(body)

        print(f"{indent}{'└── ' if last else '├── '}{' -> '.join(chain)}")
        for i, child in enumerate(children):
            print_node(child, indent + ("    " if last else "│   "), i == len(children) - 1)

    root = spec.worldbody.first_body()
    if root:
        print_node(root)


if __name__ == "__main__":
    xml = os.path.join(os.path.dirname(__file__), "humanoid_v21.xml")

    print("Body tree:")
    print_tree(xml)

    chain = get_chain(xml, 'wrist_3_L')
    print(f"\nChain: {chain}")

    model = load_partial(xml, chain)
    data = mujoco.MjData(model)
    print(f"Loaded {model.nbody} bodies, {model.njnt} joints")

    data.qpos[2] = 0.5
    data.qpos[3:7] = [1, 0, 0, 0]

    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.distance = 1.5
        while v.is_running():
            data.qpos[0:3], data.qpos[3:7], data.qvel[0:6] = [0, 0, 0.5], [1, 0, 0, 0], 0
            mujoco.mj_step(model, data)
            v.sync()
