# Set terminal and output
set terminal pngcairo enhanced

# Enable log scale for y-axis
set logscale y
set format y "10^{%T}"

# Set labels
set xlabel 'Number of tree traversals'

# Define data files and titles
files = '"liars_dice_6_rtcfr_plus_adp_7.txt" \
         "liars_dice_6_rtcfr_plus_1e-1.txt" \
         "liars_dice_6_rtcfr_plus_1e-2.txt" \
         "liars_dice_6_rtcfr_plus_1e-3.txt" \
         "liars_dice_6_rtcfr_plus_0.txt" \
         "liars_dice_6_cfr_plus_1e-3.txt" \
         "liars_dice_6_cfr_plus_0.txt" \
         "liars_dice_6_egt_1e-3.txt" \
         "liars_dice_6_egt_0.txt" \
         "liars_dice_6_regomwu_0.txt" \
         "liars_dice_6_regomwu_1e-3.txt" \
         "liars_dice_6_regomwu_efpe_adp.txt"'

titles = '"RTCFR+(adp)" \
         "RTCFR+(0.1)" \
         "RTCFR+(0.01)" \
         "RTCFR+(0.001)" \
         "RTCFR+(0)" \
         "CFR+(0.001)" \
         "CFR+(0)" \
         "EGT(0.001)" \
         "EGT(0)" \
         "RegOMWU(0)" \
         "RegOMWU(0.001)" \
         "RegOMWU(adp)"'

# Define line styles: different dash types for each algorithm, hollow points
# RTCFR+
set style line 1 lc rgb '#FF00FF' dt 1 lw 2 pt 5 ps 1
set style line 2 lc rgb '#FF8080' dt 1 lw 2 pt 9 ps 1
set style line 3 lc rgb '#FF8080' dt 1 lw 2 pt 11 ps 1
set style line 4 lc rgb '#FF8080' dt 1 lw 2 pt 13 ps 1
set style line 5 lc rgb '#FF8080' dt 1 lw 2 pt 7 ps 1

# CFR+
set style line 6 lc rgb '#0066CC' dt 2 lw 2 pt 13 ps 1
set style line 7 lc rgb '#0066CC' dt 2 lw 2 pt 7 ps 1

# EGT
set style line 8 lc rgb '#9933CC' dt 3 lw 2 pt 13 ps 1
set style line 9 lc rgb '#9933CC' dt 3 lw 2 pt 7 ps 1

# RegOMWU
set style line 10 lc rgb '#339933' dt 4 lw 2 pt 13 ps 1
set style line 11 lc rgb '#339933' dt 4 lw 2 pt 7 ps 1
set style line 12 lc rgb '#339933' dt 4 lw 2 pt 5 ps 1

# Enable legend
set key opaque bottom left

step = 5
step_point=5

set key off

# Plot exploitability
set output 'liars_dice-6-exp-full.png'
set ylabel 'Exploitability'
plot for [i=1:words(files)] word(files,i) every step_point using 2:3 with lines linestyle i title word(titles,i), \
     for [i=1:words(files)] word(files,i) every step using 2:3 with points linestyle i notitle

# Plot max information set regret
set output 'liars_dice-6-max-regret-full.png'
set ylabel 'Max information set regret'
plot for [i=1:words(files)] word(files,i) every step_point using 2:5 with lines linestyle i title word(titles,i), \
     for [i=1:words(files)] word(files,i) every step using 2:5 with points linestyle i notitle