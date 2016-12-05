library(ggplot2)

options(digits = 4)

gb = data.frame(stage = c(5000, 10000, 50000, 100000), F1=c(0.2994350282, 0.3102493074,0.3279569892,0.337801608579))

ggplot(data=gb, aes(x=stage, y=F1)) +
  geom_line(color="red")+
  geom_point() + scale_x_continuous(name = "number of boosting stages") + 
  scale_y_continuous(name="F-1 score") + 
  ggtitle("Gradient Boosting") + 
  theme(plot.title = element_text(lineheight=.8, face="bold"))