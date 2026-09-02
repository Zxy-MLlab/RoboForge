def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _point_ref(result, words):
    words = tuple(word.lower() for word in words)
    fallback = None
    for item in _walk(result):
        ref = item.get("point_ref")
        if not ref:
            continue
        fallback = fallback or ref
        label = " ".join(
            str(item.get(key, ""))
            for key in ("query", "label", "text", "name", "category")
        ).lower()
        if all(word in label for word in words):
            return ref
    if fallback is None:
        raise ValueError("perception returned no point reference")
    return fallback


def run(robot):
    frame = robot.observe(channel="rgbd", request={})
    source_perception = robot.use(
        "libero.rgbd_perception:v001",
        {
            "frame": frame,
            "queries": ["black bowl"],
            "max_detections_per_query": 8,
        },
    )
    target_perception = robot.use(
        "libero.rgbd_perception:v001",
        {
            "frame": frame,
            "queries": ["plate"],
            "max_detections_per_query": 8,
        },
    )
    source_ref = _point_ref(source_perception, ("bowl",))
    target_ref = _point_ref(target_perception, ("plate",))

    robot.act({
        "type": "gripper",
        "command": "open",
        "repeat": 12,
    })
    robot.act({
        "type": "move_to_point",
        "target_ref": source_ref,
        "offset": [0, 0, 0.12],
        "gripper": -1,
        "max_steps": 100,
        "tolerance_m": 0.01,
    })
    robot.act({
        "type": "move_to_point",
        "target_ref": source_ref,
        "offset": [0, 0, -0.015],
        "gripper": -1,
        "max_steps": 100,
        "tolerance_m": 0.008,
    })
    robot.act({
        "type": "gripper",
        "command": "close",
        "repeat": 18,
    })
    robot.act({
        "type": "move_to_point",
        "target_ref": source_ref,
        "offset": [0, 0, 0.16],
        "gripper": 1,
        "max_steps": 100,
        "tolerance_m": 0.01,
    })

    attached_frame = robot.observe(channel="rgbd", request={})
    attachment = robot.verify(
        "visual_attachment",
        {
            "frame": attached_frame,
            "object_query": "black bowl",
            "source_ref": source_ref,
        },
    )
    robot.record({"kind": "attachment_check", "result": attachment})

    robot.act({
        "type": "move_to_point",
        "target_ref": target_ref,
        "offset": [0, 0, 0.16],
        "gripper": 1,
        "max_steps": 100,
        "tolerance_m": 0.01,
    })
    robot.act({
        "type": "move_to_point",
        "target_ref": target_ref,
        "offset": [0, 0, 0.03],
        "gripper": 1,
        "max_steps": 100,
        "tolerance_m": 0.008,
    })
    robot.act({
        "type": "gripper",
        "command": "open",
        "repeat": 18,
    })
    robot.act({
        "type": "settle",
        "steps": 12,
        "gripper": -1,
    })
    robot.act({
        "type": "move_to_point",
        "target_ref": target_ref,
        "offset": [0, 0, 0.30],
        "gripper": -1,
        "max_steps": 100,
        "tolerance_m": 0.01,
    })
    final_frame = robot.observe(channel="rgbd", request={})
    verification = robot.verify(
        "visual_support_relation",
        {
            "frame": final_frame,
            "object_query": "black bowl",
            "target_query": "plate",
            "source_ref": source_ref,
            "target_ref": target_ref,
        },
    )
    robot.record({"kind": "placement_verification", "result": verification})
    return verification
