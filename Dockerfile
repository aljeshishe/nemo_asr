FROM ubuntu
RUN apt update && apt install sudo python3-pip python3-venv python3-wheel git -y
    python3 -m venv venv && . venv/bin/activate
    ./install.sh
   # cache cleanup
   apt-get clean autoclean && \
   apt-get autoremove --yes  && \
   rm -rf /var/lib/{apt,dpkg,cache,log}/

# installing g++ to fix
# "error occurred: ToolNotFound: Failed to find tool. Is `c++` installed?"
# TODO remove it from here
# disabling Please select the geographic area in which you live
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && \
    apt install curl sudo xz-utils git g++ wget gnupg2 docker.io -y && \
    # create nixbld user for installing nix
    adduser nixbld --gecos "First Last,RoomNumber,WorkPhone,HomePhone" --disabled-password && \
    echo 'nixbld ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    echo "nixbld:nixbld" | sudo chpasswd && \
    usermod -aG sudo nixbld && \
    mkdir /nix && \
    chown nixbld /nix && \
    mkdir -p /etc/nix/ && \
    # fixing error: while setting up the build environment: mounting /proc: Operation not permitted
    echo "sandbox = false" >> /etc/nix/nix.conf && \
    # install git lfs
    mkdir -p /tmp/git-lfs && \
    cd /tmp/git-lfs && \
    wget https://github.com/git-lfs/git-lfs/releases/download/v2.11.0/git-lfs-linux-amd64-v2.11.0.tar.gz && \
    tar -zxf git-lfs-linux-amd64-v2.11.0.tar.gz && \
    sudo ./install.sh && \
    cd / && \
    rm -rf /tmp/git-lfs && \
    # install git secret
    echo "deb https://dl.bintray.com/sobolevn/deb git-secret main" | tee -a /etc/apt/sources.list && \
    wget -qO - https://api.bintray.com/users/sobolevn/keys/gpg/public.key | apt-key add - && \
    apt install -y gawk git-secret && \
    # install docker
    apt install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - && \
    add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable" && \
   apt update && apt install -y docker-ce docker-ce-cli containerd.io && \
   usermod -aG docker nixbld && \
   # cache cleanup
   apt-get clean autoclean && \
   apt-get autoremove --yes  && \
   rm -rf /var/lib/{apt,dpkg,cache,log}/

USER nixbld
ENV USER nixbld
WORKDIR /home/nixbld
RUN curl -L https://nixos.org/nix/install | sh && \
    . /home/nixbld/.nix-profile/etc/profile.d/nix.sh && \
    nix-channel --add https://nixos.org/channels/nixos-20.03 nixpkgs && \
    nix-channel --update && \
    nix-env -iA cachix -f https://cachix.org/api/v1/install && \
    cachix use cprc && \
    nix-env -iA nixpkgs.bazelisk && \
    nix-env -iA nixpkgs.kubectl && \
    nix-env -iA nixpkgs.python && \
    nix-env -iA nixpkgs.python3 && \
    nix-env -iA nixpkgs.nodejs && \
    npm version && \
    # To fetch build dependencies for NIX, to be able to run Remote Build with this image
    bazelisk version && \
    sudo ln -s /home/nixbld/.nix-profile/bin/python3 /usr/bin/python && \
    touch WORKSPACE && \
    git lfs install && \
    # echo "export USER=nixbld" >> /home/nixbld/.bashrc && \
    echo '. /home/nixbld/.nix-profile/etc/profile.d/nix.sh' >> /home/nixbld/.bashrc


COPY .bazelversion .

COPY kubeconfig /home/nixbld/.kube/config
COPY docker_config.json /home/nixbld/.docker/config.json

CMD ["bash"]

