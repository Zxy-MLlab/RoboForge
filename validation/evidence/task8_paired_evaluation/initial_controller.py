def run(robot):
    frame = robot.observe(channel="rgbd", request={})
    perception = robot.use(
        "libero.rgbd_perception:v001",
        {
            "frame": frame,
            "queries": ["black bowl", "plate"],
            "distinct_query_pairs": [["black bowl", "plate"]],
            "max_detections_per_query": 8,
        },
    )
    robot.record({"kind": "perception", "result": perception})
    return perception
