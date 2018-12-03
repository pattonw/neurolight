#!/bin/bash

SINGULARITY_LOCALCACHEDIR=/root/tmp
SINGULARITY_TMPDIR=/root/tmp
TEMPDIR=/root/tmp
SINGULARITY_PULLFOLDER=/root/tmp

export SINGULARITY_LOCALCACHEDIR
export SINGULARITY_TMPDIR
export SINGULARITY_PULLFOLDER
export TEMPDIR

sudo singularity build neurolight:v0.1.img Singularity
