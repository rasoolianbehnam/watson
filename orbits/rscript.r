library('orca')
for (i in 0:72) {
    filename = sprintf("orbit%02d.txt", i);
    a <- read.csv(filename, header=FALSE); 
    b = count5(a); 
    c = sum(b[1, ] > 0); 
    print(c);
}
