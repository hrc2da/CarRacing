#%%
library("lme4")
library("ggplot2")
library("sjPlot")
# design_data <- read.csv("chopshopstudyscripts/human_design_test_drives.csv")
h1_design_data <- read.csv("chopshopstudyscripts/data/h1_delta_test_drives.csv")
# model1 <- lmer(formula = reward ~ 1 + eng_power + friction_lim + wheel_rad #(1|session), data= design_data)



h1_mixed_model <- lmer( reward ~ 1 + (1|session) + (1|map_num), h1_design_data)
h1_sessions_bpl = ggplot(h1_design_data, aes(x=factor(session), y=reward), labels=c(1,2,3,4,5,6,7,8,9,10,11,12)) + geom_hline(yintercept=0, linetype="dashed", color = "red", size=2) +
    geom_boxplot() + xlab("Participant") + ylab("Reward Improvement vs. Original Car") +
    theme(text = element_text(size=20),axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) + coord_cartesian(ylim = c(-500,1000))

ggsave(filename='chopshopstudyscripts/plots/h1_sessions_bpl.png',plot=h1_sessions_bpl,width=10,height=8)

h2_design_data <- read.csv("chopshopstudyscripts/data/h2_delta_test_drives.csv")
# model1 <- lmer(formula = reward ~ 1 + eng_power + friction_lim + wheel_rad #(1|session), data= design_data)



h2_mixed_model <- lmer( reward ~ 1 + (1|session) + (1|map_num), h2_design_data)
h2_sessions_bpl = ggplot(h2_design_data, aes(x=factor(session), y=reward), labels=c(1,2,3,4,5,6,7,8,9,10,11,12)) + geom_hline(yintercept=0, linetype="dashed", color = "red", size=2) +
    geom_boxplot() + xlab("Participant") + ylab("Reward Improvement vs. Original Car") +
    theme(text = element_text(size=20),axis.text.x = element_text(size=20),axis.text.y = element_text(size=20)) + coord_cartesian(ylim = c(-500,1000))

ggsave(filename='chopshopstudyscripts/plots/h2_sessions_bpl.png',plot=h2_sessions_bpl,width=10,height=8)


# varlist = names(design_data)[2:(length(design_data)-1)]


# Formula1 <- formula(paste("reward ~ ", paste(varlist, collapse=" + " ))) 

# model1 <- lm(Formula1, design_data)

# Formula2 <- formula(paste("reward ~ ", paste(varlist, collapse=" + " ), " + (1|session)")) 

# mixed_model <- lmer(Formula2, design_data)
