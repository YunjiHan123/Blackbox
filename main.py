from core.pipeline import create_pipeline
from core.runtime import run_camera_runtime


def main():

    pipeline = create_pipeline()
    run_camera_runtime(pipeline)


if __name__ == "__main__":
    main()
