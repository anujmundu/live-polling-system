FROM grafana/grafana:10.2.2
COPY monitoring/grafana/provisioning/ /etc/grafana/provisioning/
