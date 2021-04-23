data {
	int K;
	real s[K, 2];
	real d[K];
}

parameters {
	real<lower=0> sigma;
	real t[2];
}

model {
	real sq_dist;

	sigma ~ cauchy(0, 1);

	for (k in 1:K) {
		sq_dist = 0;

		for (dim in 1:2) {
			sq_dist += (t[dim] - s[k, dim])^2;
		}
		
		d[k] ~ normal(sqrt(sq_dist), sigma);

	}

}



