dat=rnorm(300)
hist(dat,freq=F,main="频率分布直方图与概率密度函数")
mtext("样本容量300",side=3,line=0)
lines(density(dat,bw=0.5),col='red',lwd=3)
z=sample(1:1,300,replace=TRUE)
plot(sort(dat),cumsum(z/300),main = "正态分布的频率阶梯函数")
mtext("样本容量300",side=3,line=0)
lines(sort(dat),cumsum(z/300),col='blue',lwd=2)

