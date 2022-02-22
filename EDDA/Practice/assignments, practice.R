#assignment 1

#Q2
n=m=30
mu=180
sd=15
B=100
B=1000
nus=seq(175,185,by=0.25)
p=numeric(B); C=length(nus); powers=numeric(C);

for (c in 1 : C){
  nu = nus[c]
  for (b in 1:B){
  x=rnorm(m,mu,sd); y=rnorm(n,nu,sd);
  p[b]=t.test(x,y,var.equal=TRUE)[[3]]}
  powers[c]=mean(p<0.05)}
plot(nus,powers,type="l",xlab="nu",ylab="power")

data = read.table(file="telephone.txt", header = TRUE)
boxplot(data$Bills)
hist(data$Bills) #included 0
data_zero = data[apply(data, 1, function(row) all(row !=0 )), ]
data_zero


#Q4

run=read.table(file="run.txt", header = TRUE)
run
qdata=run$before+run$after
qqnorm(run$before)
qqnorm(run$after)
t.test(run$before[1:12],run$after[1:12], paired=TRUE)
t.test(run$before[13:24],run$after[13:24], paired=TRUE)
diflemo = (run$before[1:12]-run$after[1:12])
difenergy = run$before[13:24]-run$after[13:24]
shapiro.test(run$before[1:12])
shapiro.test(run$before[13:24])
shapiro.test(run$after[1:12])
shapiro.test(run$after[13:24])
qqnorm(run$before[1:12])
wilcox.test(diflemo, difenergy)

#Q5
qqnorm(chickwts$weight)
shapiro.test(chickwts$weight)
boxplot(chickwts$weight)
#chickframe = data.frame(yield=as.vector(chickwts$weight),feed=as.factor(chickwts$feed))
#chickframe
chickaov=lm(chickwts$weight~chickwts$feed, data=chickwts)
anova(chickaov)
summary(chickaov)
qqnorm(chickwts$weight)
shapiro.test(chickwts$weight) #normally dist
qqnorm(residuals(chickaov))
shapiro.test(residuals(chickaov))
#residuals vs fitted
plot(fitted(chickaov), residuals(chickaov)) #no pattern must be seen
kruskal.test(chickwts$weight,chickwts$feed,data=chickwts)
boxplot(chickwts$weight~chickwts$feed, data=chickwts)



#Assignment2
bread=read.table(file="bread.txt")
bread
N=3; I=2; J=3
rbind(rep(1:I,each=N*J), rep(1:J,N*I), sample(1:(N*I*J)))
par(mfrow=c(1,2))
boxplot(bread$hours~bread$environment)
boxplot(bread$hours~bread$humidity)

environment=as.factor(bread$environment); humidity=as.factor(bread$humidity)
breadaov=lm(bread$hours~environment*humidity, data=bread)
anova(breadaov)
summary(breadaov)
par(mfrow=c(1,1))
qqnorm(breadaov$residuals)
shapiro.test(breadaov$residuals)
cooks=round(cooks.distance(breadaov), 2)
plot(cooks)

#search engine

N=5; I=3; J=1 
for (i in 1:N) {
cat('Skill:', i, '-', sample(1:(J*I)) + 3*(i-1), '\n')}

search=read.table(file="search.txt")
search
is.factor(search$interface)
is.factor(skill)
search$interface=as.factor(search$interface)
search$skill=as.factor(search$skill)
searchaov=lm(search$time~search$skill+search$interface, data=search)
anova(searchaov)
summary(searchaov)

friedman.test(search$time,search$interface,search$skill,data=search)
searchaov2=lm(search$time~search$interface, data=search)
summary(searchaov2)



cow = read.table(file="cow.txt")
cow
cow$per=as.factor(cow$per)
cow$id=as.factor(cow$id)
cowaov=lmer(cow$milk~cow$order+cow$per+cow$X..treatment.+(1|id),data=cow)
summary(cowaov)
library(lme4)
cowlmer1=lmer(milk~order+per+(1|id),data=cow,REML=FALSE)
anova(cowaov,cowlmer1)
t.test(cow$milk[cow$X..treatment.=="A"],cow$milk[cow$X..treatment.=="B"],paired=TRUE)

austen=read.table(file ="austen.txt")
austen2=read.table(file ="austen.txt")
austen2$Sand2=NULL
austen2
z=chisq.test(austen2); z

#werkt-niet
crime=read.table(file="expensescrime.txt")
crime
summary(lm(expend~bad+crime+lawyers+employ+pop, data=expensescrime))

pairs(crime)


#Assignment3

fruitflies
fruitflies$loglongevity = log(fruitflies$longevity)
fruitflies
boxplot(fruitflies$loglongevity~fruitflies$activity)
fruitaov=lm(loglongevity~activity, data=fruitflies)
summary(fruitaov) #e^intercept is back to days
anova(fruitaov)

fruitaov2=lm(loglongevity~thorax+activity, data=fruitflies)
summary(fruitaov2)
av_thorax=mean(fruitflies$thorax)

fruitaov3=lm(loglongevity~thorax, data=fruitflies)
summary(fruitaov3)
plot(fruitflies$loglongevity~fruitflies$thorax, data = fruitflies, pch=unclass(activity))
for (i in 1:3) abline(lm(loglongevity~thorax,data=fruitflies[as.numeric(fruitflies$activity)==i,]))
fruitaov4=lm(loglongevity~activity*thorax, data=fruitflies)
summary(fruitaov4)


#titanic
titanic$Survived=as.numeric(titanic$Survived)
titanic$Survived



#africa
africa$pollib=as.factor(africa$pollib)
afglm=glm(miltcoup~oligarchy+pollib+parties+pctvote+popn+size+numelec+numregim,family=poisson,data=africa)
summary(afglm)
afglm1=glm(miltcoup~oligarchy+pollib+parties,family=poisson,data=africa)
summary(afglm1)
drop1(afglm1, test="Chisq")


treeaov = lm(treeVolume$volume~treeVolume$type, data=treeVolume)
summary(treeaov)

treelm=lm(volume~diameter+height+type,data=treeVolume)
summary(treelm)
volume_beech = (4.69806*mean(treeVolume$diameter)) + (0.41725*mean(treeVolume$height)) - 63.78138 
volume_oak =   (4.69806*mean(treeVolume$diameter)) + (0.41725*mean(treeVolume$height)) - 63.78138 - 1.30460
oak=data.frame(diameter = mean(treeVolume$diameter), height = mean(treeVolume$height), type="oak")
predict(treelm,  oak, type="response")

#A
diet = read.table(file="diet.txt", header = TRUE)
x = diet$preweight
y = diet$weight6weeks
t.test(x,y,paired = TRUE)
qqnorm(x); hist(x)
shapiro.test(x) #p>0.05, normally distributed
qqnorm(y); hist(y) #p<0.05, normally distributed
shapiro.test(y) 
#so we use wilcox
wilcox.test(x, y, paired = TRUE) #p<0.05

#B
diet$weight.lost = diet$preweight-diet$weight6weeks
n = length(diet$weight.lost); w = sum(diet$weight.lost>3)
binom.test(w,n,0.5,alternative = "g")

#C
diet$diet = as.factor(diet$diet)
diet1 = lm(weight.lost~diet, data = diet)
anova(diet1)
summary(diet1)
qqnorm(residuals(diet1)) #looks normal
shapiro.test(residuals(diet1)) #p>0.05, so residuals are normally distributed
plot(fitted(diet1), residuals(diet1)) #looks ok, no pattern
#

#D
diet$gender = as.factor(diet$gender)
diet2 = lm(weight.lost ~ diet*gender, data=diet)
anova(diet2)
summary((diet2))
#there is a slight interaction between diet:gender.
#gender is not significant


#E
diet3 = lm(weight.lost ~ height*diet, data = diet)
anova(diet3) #no interaction (p>0.05). diet is has significant influence.
summary(diet3)
diet4 = lm(weight.lost ~ diet + height, data=diet)
anova(diet4)
drop1(diet4, test="F")

#F
av_person = mean(diet$preweight)
dietfram1=data.frame(av_person, diet="1")
dietfram2=data.frame(av_person, diet="2")
dietfram3=data.frame(av_person, diet="3")
predict(diet1, dietfram1, type="response")
predict(diet1, dietfram2, type="response")
predict(diet1, dietfram3, type="response")

#G
diet$lost.4kg = diet$weight.lost>4
diet$lost.4kg = as.numeric(diet$lost.4kg)
dietglm = glm(lost.4kg ~ diet, data=diet, family = binomial)
anova(dietglm, test="Chisq")

diet$lost.4kg = as.factor(diet$lost.4kg)
dietglm = glm(lost.4kg ~ diet, data=diet, family = binomial)
anova(dietglm, test="Chisq")






