# 自适应裁切 + padding + 增加legend间距
set terminal pngcairo enhanced size 800,400 crop
set output 'legend.png'

# 定义线型
set style line 1 lc rgb '#FF00FF' dt 1 lw 2 pt 5 ps 1
set style line 2 lc rgb '#FF8080' dt 1 lw 2 pt 9 ps 1
set style line 3 lc rgb '#FF8080' dt 1 lw 2 pt 11 ps 1
set style line 4 lc rgb '#FF8080' dt 1 lw 2 pt 13 ps 1
set style line 5 lc rgb '#FF8080' dt 1 lw 2 pt 7 ps 1
set style line 6 lc rgb '#0066CC' dt 2 lw 2 pt 13 ps 1
set style line 7 lc rgb '#0066CC' dt 2 lw 2 pt 7 ps 1
set style line 8 lc rgb '#9933CC' dt 3 lw 2 pt 13 ps 1
set style line 9 lc rgb '#9933CC' dt 3 lw 2 pt 7 ps 1
set style line 10 lc rgb '#339933' dt 4 lw 2 pt 13 ps 1
set style line 11 lc rgb '#339933' dt 4 lw 2 pt 7 ps 1
set style line 12 lc rgb '#339933' dt 4 lw 2 pt 5 ps 1

# 图例标题
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

# 去掉坐标轴
unset border
unset tics
unset xlabel
unset ylabel

# 范围稍大，确保有padding
set xrange [-1:1]
set yrange [-1:1]
set offsets 0.3,0.3,0.3,0.3

# 图例设置：居中、自动排列、加间距
set key center center maxrows 4 maxcols 3 box opaque
set key spacing 1.5    # 垂直间距倍数（默认 1.0）
set key samplen 2.5    # 符号与文字的水平间距

# 绘制虚拟点
plot for [i=1:12] '+' using (0):(0) with linespoints ls i title word(titles,i)

unset output
