library("orca")
args <- commandArgs(trailingOnly = TRUE)
chr <- as.integer(args[1])

mit_file_name <- sprintf("../data/chr%02d_chr%02d_mit.edges", chr, chr)
all_file_name <- sprintf("../data/chr%02d_chr%02d_all.edges", chr, chr)
rl_file_name <- sprintf("../data/chr%02d_chr%02d_rl.edges", chr, chr)
call4_file_name <- sprintf("../data/chr%02d_chr%02d_call4.edges", chr, chr)

print(mit_file_name)
print(all_file_name)
print(rl_file_name)
print(call4_file_name)

mit_file <- read.csv(mit_file_name)
all_file <- read.csv(all_file_name)
rl_file <- read.csv(rl_file_name)
call4_file <- read.csv(call4_file_name)

mit_graphlets <- count5(mit_file)
all_graphlets <- count5(all_file)
rl_graphlets <- count5(rl_file)
call4_graphlets <- count5(call4_file)

mit_graphlet_file = sprintf("../data/chr%02d_chr%02d_mit.graphlets", chr, chr)
all_graphlet_file = sprintf("../data/chr%02d_chr%02d_all.graphlets", chr, chr)
rl_graphlet_file = sprintf("../data/chr%02d_chr%02d_rl.graphlets", chr, chr)
call4_graphlet_file = sprintf("../data/chr%02d_chr%02d_call4.graphlets", chr, chr)

write.table(mit_graphlets, file = mit_graphlet_file ,row.names=FALSE, col.names=FALSE)
write.table(all_graphlets, file = all_graphlet_file ,row.names=FALSE, col.names=FALSE)
write.table(rl_graphlets, file = rl_graphlet_file ,row.names=FALSE, col.names=FALSE)
write.table(call4_graphlets, file = call4_graphlet_file ,row.names=FALSE, col.names=FALSE)
