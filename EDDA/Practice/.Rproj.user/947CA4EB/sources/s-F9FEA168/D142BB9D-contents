#ass2, Q3,4,5

#Q3
cow$id=as.factor(cow$id); cow$per=as.factor(cow$per)
cowlm=lm(milk~treatment+id+order+per, data=cow)
anova(cowlm)
summary(cowlm)
library(lme4)
cowlm2=lmer(milk~order+per+treatment+(1|id), data=cow, REML=FALSE)
cowlm3=lmer(milk~order+per+(1|id), data=cow, REML=FALSE)
anova(cowlm3,cowlm2)

attach(cow)
t.test(milk[treatment=="A"],milk[treatment=="B"], paired=TRUE)

#Q4


#Q5
#A
attach(expensescrime)
pairs(expensescrime)
crime2=expensescrime
crime2$state=NULL
cor(crime2)
crimelm=lm(expend~bad+crime+lawyers+employ+pop,data=crime2)
order(abs(residuals(crimelm)))
qqnorm(residuals(crimelm))
plot(fitted(crimelm), residuals(crimelm))
round(cooks.distance(crimelm),2)
plot(1:51,cooks.distance(crimelm),type="b")
#data points 5,8,35 and 44 must be removed as they cross the 1.0 cooks distance
#mark and ar considered as outliers.


#remove outliers!!
crime3=crime2[-c(5,8,35,44), ]
crime3

#without outliers
crimelm2=lm(expend~bad+crime+lawyers+employ+pop,data=crime3)
order(abs(residuals(crimelm2)))
qqnorm(residuals(crimelm2))
plot(fitted(crimelm2), residuals(crimelm2))
round(cooks.distance(crimelm2),2)
plot(1:47,cooks.distance(crimelm2),type="b") #all cooksdistances below 1



#B
#step-up:
attach(expensescrime)
summary(lm(expend~bad, date=expensescrime))
summary(lm(expend~crime, date=expensescrime))
summary(lm(expend~lawyers, date=expensescrime))
summary(lm(expend~employ, date=expensescrime))
summary(lm(expend~pop, date=expensescrime))

summary(lm(expend~employ+bad, date=expensescrime))
summary(lm(expend~employ+crime, date=expensescrime))
summary(lm(expend~employ+lawyers, date=expensescrime))
summary(lm(expend~employ+pop, date=expensescrime))

summary(lm(expend~employ+lawyers+bad, date=expensescrime))
summary(lm(expend~employ+lawyers+expensescrime$crime, date=expensescrime))
summary(lm(expend~employ+lawyers+pop, date=expensescrime))
#nothing else significant: so resulting model is:
step_up_model=lm(expend~employ+lawyers, date=expensescrime)
summary(step_up_model)
#step-down:
step_down_model=lm(expend~lawyers+employ, data=expensescrime)

#diagnostic tests
par(mfrow=c(1,2))
plot(fitted(step_up_model),residuals(step_up_model))
qqnorm(residuals(step_up_model))
shapiro.test(residuals(step_up_model))

plot(fitted(step_down_model),residuals(step_down_model))
qqnorm(residuals(step_down_model))
shapiro.test(residuals(step_down_model))





x = run$before; y=run$after
qqnorm(x); qqnorm(y)
shapiro.test(x)
shapiro.test(y)
cor.test(x,y) #significant, there is correlation

before_lemo = run$before[1:12]; after_lemo = run$after[1:12]
par(mfrow=c(1,2))
qqnorm(before_lemo); qqnorm(after_lemo)
shapiro.test(before_lemo); shapiro.test(after_lemo) #normally distributed

run$difference = run$before - run$after
dif_lemo = run$difference[1:12]
dif_energy = run$difference[13:24]
qqnorm(dif_lemo); qqnorm(dif_energy)
shapiro.test(dif_lemo); shapiro.test(dif_energy)

wilcox.test(dif_energy, dif_lemo)
ks.test(dif_energy, dif_lemo)

par(mfrow=c(1,2))
boxplot(bread$hours~bread$environment)
boxplot(bread$hours~bread$humidity)
interaction.plot(bread$humidity, bread$environment, bread$hour)
interaction.plot(bread$environment,bread$humidity, bread$hour)


breadlm=lm(hours~humidity*environment, data=bread)
anova(breadlm)
drop1(breadlm, test="F")
summary(breadlm)
plot(fitted(breadlm), residuals(breadlm))
qqnorm(residuals(breadlm))
shapiro.test(residuals(breadlm))

search$skill = as.factor(search$skill)
search$interface = as.factor(search$interface)
searchlm = lm(time ~ skill+interface, data=search)
summary(searchlm)
anova(searchlm)
skillint = data.frame(skill = '3', interface = '3')
predict(searchlm, skillint, type ='response')
friedman.test(search$time,search$interface,search$skill)

only_int = lm(time~interface, data=search)
anova(only_int)
cow = read.table(file="cow.txt", header = T)
cow
milk = lm(milk~order+id+per+treatment, data = cow)
anova(milk)
summary(milk)

milk2 = lmer(milk~order+per+treatment+(1|id), data = cow)
summary(milk2)
milk3 = lmer(milk~order+per+(1|id), data = cow)
anova(milk3, milk2)



austen = read.table(file="austen.txt", header = T)
austen_seff = austen[0:3]
austen_seff
z = chisq.test(austen_seff)
residuals(z) # = (z$observed-z$expected)/sqrt(z$expected)
austen

z=chisq.test(austen)
z
residuals(z)




crime = read.table(file = "expensescrime.txt", header = T)
crime

plot(crime[2:7])
crimelm = lm(expend~bad+crime+lawyers+employ+pop, data=crime)
install.packages("car")
vif(crimelm)
cooks = cooks.distance(crimelm)
plot(cooks)

summary(crimelm)
anova(crimelm)
crime1 = lm(expend~employ+lawyers, data=crime)
summary(crime1)


fruitflies = read.table(file = "fruitflies.txt", header=T)
fruitflies$loglongevity = log(fruitflies$longevity)
fruitflies
boxplot(fruitflies$loglongevity~fruitflies$activity)
plot(fruitflies$loglongevity~fruitflies$thorax, data = fruitflies, pch=unclass(activity))
for (i in 1:3) abline(lm(loglongevity~thorax,data=fruitflies[as.numeric(fruitflies$activity)==i,]))
fruitaov4=lm(loglongevity~activity, data=fruitflies)
summary(fruitaov4)
fruitaov5=lm(loglongevity~thorax+activity, data=fruitflies)
drop1(fruitaov5, test="F")
summary(fruitaov5)
fruitaov6=lm(loglongevity~thorax*activity, data=fruitflies)
anova(fruitaov6)
summary(fruitaov6)

titanic = read.table(file="titanic.txt", header = T)
titglm = glm(Survived~PClass+Age+Sex, family = binomial, data=titanic)
anova(titglm, test="Chisq")
anova(titglm, test="Chisq")
titglm2 = glm(Survived~PClass*Age, family = binomial, data=titanic)
anova(titglm2, test="Chisq")
titglm3 = glm(Survived~Sex*Age, family = binomial, data=titanic)
anova(titglm3, test="Chisq")

titglm4 = glm(Survived~Sex*Age + PClass, family = binomial, data=titanic)
anova(titglm4, test = "Chisq")
titglm5 = glm(Survived~Age*PClass + Sex, family = binomial, data=titanic)
anova(titglm5, test = "Chisq")


