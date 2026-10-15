import sys
import pyray as pr
import prepare_textures as prep
from user_interface import run_entire_world, parse_attributes
import numpy as np
from qiskit_aer import AerSimulator

print("Welcome to the quantum terrain generator!")
print("-----------------------------------------")
world = None
path = None
seed = None
worldsize = None

choice = input(
    "Do you want to load (L) a file or simulate (S) fresh data? \n Type S or L then enter...\n"
)
if choice not in ["S", "L"]:
    print("invalid choice, try again...")
    sys.exit(0)
if choice == "L":
    path = input(
        "Please type the filepath then press enter...\n The file should contain 8x8 chunks\n"
    )
    try:
        prep.prepare_assets_from_file(path, 4, (1 + np.sqrt(5)) / 2)
        attr = parse_attributes(path)
        seed = attr["seed"]
        worldsize = attr["max. chunks"]
    except OSError:
        print("that was not a valid file path!")
        sys.exit(0)
    except KeyError:
        print("this file was probably not a good world file!")
        sys.exit(0)
elif choice == "S":
    try:
        worldsize = int(
            input(
                "Please choose a world size. For 'n' this will be n by n chunks of 16x16 pixels... Powers of two required, max of 8.\n "
            )
        )
        if worldsize > 8 or worldsize < 0 or worldsize not in [1, 2, 4, 8]:
            raise ValueError("world too large! (or less than zero)")
        seed = int(input("Please choose a seed between 0 and 2^n-1\n"))

        if seed >= 2**worldsize or seed < 0:
            raise ValueError("seed was too large or less than zero")
        pcg_params = [0, 0]
        m = input(
            "You may choose parameters for the PCG (not recommended). Press enter to skip. Otherwise enter multiplier, then enter...\n"
        )
        c = input("Enter PCG additive offset or skip...\n")

        if m == "":
            pcg_params[0] = 21
        else:
            pcg_params[0] = int(m)
        if c == "":
            pcg_params[1] = 3
        else:
            pcg_params[1] = int(c)

        print("Simulating the world... This may take 3-4 hours for world size 8...")
        world = run_entire_world(
            seed,
            worldsize,
            AerSimulator(method="matrix_product_state"),
            2**16,
            pcg_params=pcg_params,
        )
        prep.prepare_assets_from_world(
            world, int(np.log2(worldsize)) + 1, (1 + np.sqrt(5)) / 2, save_map=True
        )
    except ValueError:
        print("That was not a valid number or did not meet the requirements!")
        sys.exit(0)


width = 1000
height = 1000

pr.init_window(width, height, "Quantum Terrain!")
cam = pr.Camera3D()
cam.position = [42, 42, 42]
cam.target = [0, 0, 0]
cam.up = [0, 1, 0]
cam.fovy = 45
cam.projection = pr.CameraProjection.CAMERA_PERSPECTIVE


elev = pr.load_image("heightmap.png")
color = pr.load_image("colors.png")

tex = pr.load_texture_from_image(color)

mesh = pr.gen_mesh_heightmap(elev, [32, 12, 32])
model = pr.load_model_from_mesh(mesh)
model.materials[0].maps[pr.MATERIAL_MAP_DIFFUSE].texture = tex
mapPos = [-16, 0, -16]
pr.set_target_fps(60)

while not pr.window_should_close():
    pr.update_camera(cam, pr.CameraMode.CAMERA_THIRD_PERSON)
    pr.begin_drawing()
    pr.clear_background(pr.LIGHTGRAY)
    pr.begin_mode_3d(cam)
    pr.draw_model(model, mapPos, 1, pr.WHITE)
    pr.draw_grid(38, 1)
    pr.end_mode_3d()
    pr.draw_fps(10, 10)
    msg = f"SEED: {seed}, WORLD SIZE: {worldsize}"
    length = pr.measure_text(msg, 24)
    pr.draw_text(msg, width - 10 - length, 10, 24, pr.BLUE)
    pr.end_drawing()

pr.unload_texture(tex)
pr.unload_model(model)
pr.close_window()
