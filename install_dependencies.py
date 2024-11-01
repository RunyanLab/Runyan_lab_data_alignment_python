import subprocess

# Define the environment name and dependencies
env_name = "your_env_name"  # Replace with your desired environment name

dependencies = [
    "fmt=9.1.0",
    "fonttools=4.25.0",
    "freetype=2.12.1",
    "frozenlist=1.4.0",
    "fsspec=2023.10.0",
    "future=0.18.3",
    "gensim=4.3.0",
    "gettext=0.21.0",
    "gflags=2.2.2",
    "giflib=5.2.1",
    "gitdb=4.0.7",
    "gitpython=3.1.37",
    # ... (other dependencies)
]

pip_dependencies = [
    "absl-py==2.1.0",
    "astunparse==1.6.3",
    "autograd==1.6.2",
    "cython==3.0.9",
    "flatbuffers==24.3.7",
    "frozendict==2.4.0",
    "gast==0.5.4",
    "gdown==5.2.0",
    "google-pasta==0.2.0",
    "grpcio==1.62.1",
    "gym==0.24.1",
    "gym-notices==0.0.8",
    "h5py==3.10.0",
    "hdf5storage==0.1.19",
    "html5lib==1.1",
    "keras==3.1.1",
    "libclang==18.1.1",
    "ml-dtypes==0.3.2",
    "multitasking==0.0.11",
    "namex==0.0.7",
    "natsort==8.4.0",
    "neo==0.13.4",
    "oasis-deconv==0.2.0",
    "opt-einsum==3.3.0",
    "optree==0.11.0",
    "peewee==3.17.1",
    "pypdf2==3.0.1",
    "quantities==0.16.1",
    "rastermap==0.9.5",
    "slicetca==1.0.4",
    "tensorboard==2.16.2",
    "tensorboard-data-server==0.7.2",
    "tensorflow==2.16.1",
    "tensorflow-io-gcs-filesystem==0.36.0",
    "tensorly==0.8.1",
    "termcolor==2.4.0",
    "yf==0.0.4",
    "yfinance==0.2.37"
]

def main():
    # Step 1: Create the Conda environment with the specified Python version
    print(f"Creating the Conda environment: {env_name}")
    subprocess.run(["conda", "create", "-n", env_name, "python=3.11", "-y"], check=True)

    # Step 2: Install dependencies using Conda
    print(f"Installing dependencies in {env_name}")
    for dep in dependencies:
        try:
            subprocess.run(["conda", "install", "-n", env_name, dep, "-y"], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}. Skipping...")

    # Step 3: Install pip dependencies
    if pip_dependencies:
        print("Installing pip dependencies")
        for dep in pip_dependencies:
            try:
                subprocess.run(["conda", "run", "-n", env_name, "pip", "install", dep], check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to install {dep}. Skipping...")

    print(f"Environment {env_name} setup complete.")

if __name__ == "__main__":
    main()
