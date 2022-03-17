from objects import Vector3D, PI, Planet, Simulation
from ursina.shaders import lit_with_shadows_shader
from ursina import *
import typing as tp
import time


SIM: Simulation
START = time.perf_counter()
TIME_SCALE = 1

# sim settings
PAUSE = True
WAS_LAST_HELD: tp.List[str] = []

def update() -> None:
    """
    called by ursina
    """
    global START, PAUSE
    now = time.perf_counter()
    dt = now-START
    START = now

    if held_keys["down arrow"]:
        camera.rotation_x += 1

    if held_keys["up arrow"]:
        camera.rotation_x -= 1

    if held_keys["left arrow"]:
        camera.rotation_y -= 1

    if held_keys["right arrow"]:
        camera.rotation_y += 1

    if held_keys["w"]:
        camera.y += .1

    if held_keys["s"]:
        camera.y -= .1

    if held_keys["a"]:
        camera.x -= .1

    if held_keys["d"]:
        camera.x += .1

    if held_keys["p"]:
        if not "p" in WAS_LAST_HELD:
            PAUSE = not PAUSE
            WAS_LAST_HELD.append("p")

    else:
        if "p" in WAS_LAST_HELD:
            WAS_LAST_HELD.remove("p")

    if not PAUSE:
        SIM.iter(dt*TIME_SCALE)


def main() -> None:
    """
    main program
    """
    global SIM
    app = Ursina()
    EditorCamera()

    # configure window
    window.title = '3dGravitySim'
    window.borderless = True
    window.fullscreen = True
    window.exit_button.visible = False
    window.fps_counter.enabled = True
    window.color=(0, 0, 0, 0)

    # stuff
    SIM = Simulation([
        Planet(name="1", diameter=0.5, mass=2, position=Vector3D.from_cartesian(-1, 0, 0),
               velocity=Vector3D.from_polar(angle_xy=0, angle_xz=0, length=1)),

        Planet(name="2", diameter=0.5, mass=2, position=Vector3D.from_cartesian(0, 0, 0)),
        Planet(name="2", diameter=0.5, mass=2, position=Vector3D.from_cartesian(0.51, 0, 0)),
        Planet(name="2", diameter=0.5, mass=2, position=Vector3D.from_cartesian(1.02, 0, 0))
    ])
    # setup camera
    camera.x = 0
    camera.y = 2
    camera.z = 0

    # create floor plane
    Entity(model='plane', scale=10, color=color.black,
           x=0, y=-1    , z=0,
           shader=lit_with_shadows_shader)

    # create lighting
    pivot = Entity()
    DirectionalLight(parent=pivot, y=2, z=3, shadows=True, rotation=(45, -45, 45))

    # load coordinate system file
    # sys = load_model("coord_sys", path=Path("/home/nilusink/Documents/assets/simple_coord_system.obj"))
    # Entity(model=sys,
    #        x=0,
    #        y=0,
    #        z=0,
    #        origin=(0, 0))

    # start app
    app.run()


if __name__ == "__main__":
    main()