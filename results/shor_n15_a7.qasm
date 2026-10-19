OPENQASM 2.0;
include "qelib1.inc";
// Shor's Period-Finding Circuit for N = 15, a = 7 (period r = 4)
// Counting register: q[0] (LSB, weight 2^0), q[1] (weight 2^1), q[2] (MSB, weight 2^2)
// Work register: q[3], q[4] encoding {7^0=1, 7^1=7, 7^2=4, 7^3=13} mod 15
qreg q[5];
creg c[3];

h q[0];
h q[1];
h q[2];

// Controlled-U^{2^0}: cyclic shift +1 mod 4 controlled on q[0]
ccx q[0], q[3], q[4];
cx q[0], q[3];

// Controlled-U^{2^1}: cyclic shift +2 mod 4 controlled on q[1]
cx q[1], q[4];

// 3-Qubit Inverse Quantum Fourier Transform (IQFT_3) on q[0..2]
swap q[0], q[2];
h q[0];
cu1(-pi/2) q[0], q[1];
h q[1];
cu1(-pi/4) q[0], q[2];
cu1(-pi/2) q[1], q[2];
h q[2];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
