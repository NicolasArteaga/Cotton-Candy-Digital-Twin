#!/bin/bash
sudo systemctl restart cc_environment.service
sudo systemctl restart cc_plug.service
sudo systemctl restart cc_weigh_touch.service
sudo systemctl restart cc_weigh_max.service
sudo systemctl restart cc_weigh_single.service