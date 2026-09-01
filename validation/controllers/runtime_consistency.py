def run(robot):
    return {"observation": robot.observe(channel="rgbd", request={})}
