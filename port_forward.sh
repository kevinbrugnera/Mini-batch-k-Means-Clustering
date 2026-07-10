ssh -C -N -f -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
  -L 18080:10.67.22.135:8080 \
  -L 14040:10.67.22.135:4040 \
  -L 19080:10.67.22.135:18080 \
  rferrant@gate.cloudveneto.it
