FROM prom/prometheus:v2.48.0
COPY monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
