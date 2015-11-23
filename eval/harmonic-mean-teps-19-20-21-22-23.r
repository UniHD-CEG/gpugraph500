id__19__84747__harmonic_mean_TEPS <- 1.680e+08

id__19__84747__total_gpus <- 16

vect__19 <- c(get(paste('id__19__84747__','harmonic_mean_TEPS',sep='')))

labels_x <- labels_x <- c('TEPS')

id__20__84746__harmonic_mean_TEPS <- 1.977e+08

id__20__84746__total_gpus <- 16

vect__20 <- c(get(paste('id__20__84746__','harmonic_mean_TEPS',sep='')))

id__21__84744__harmonic_mean_TEPS <- 2.160e+08

id__21__84744__total_gpus <- 16

vect__21 <- c(get(paste('id__21__84744__','harmonic_mean_TEPS',sep='')))

id__22__31694__harmonic_mean_TEPS <- 2.334e+08

id__22__31694__total_gpus <- 16

vect__22 <- c(get(paste('id__22__31694__','harmonic_mean_TEPS',sep='')))

id__23__31691__harmonic_mean_TEPS <- 2.516e+08

id__23__31691__total_gpus <- 16

vect__23 <- c(get(paste('id__23__31691__','harmonic_mean_TEPS',sep='')))

labels_y <- c("sf19","sf20","sf21","sf22","sf23")

mat <- matrix(c(vect__19,vect__20,vect__21,vect__22,vect__23), nrow = 5, ncol = length(labels_x), byrow = TRUE, dimnames = list(labels_y, labels_x))

par(xpd = TRUE)
    matplot(mat, axes = FALSE, xlab='scale factor (#)', ylab = 'edges per second (s)',
        main = 'Execution on Fermi. Scale factor comparison', col = topo.colors(length(rownames(mat)), alpha=1),
        type='o', lty=1, lwd=1, pch=3)
        axis(1, at = 1:length(colnames(t(mat))), labels= labels_y)
        axis(2)
        legend('top', fill = topo.colors(length(rownames(mat)), alpha=1),
        legend = labels_x)

