#!/bin/bash

# Color
Yellow='\033[1;33m'
GREEN='\033[92m'
NC='\033[0m' # No Color

# Get correct path for this file and cd into it -> getting correct relevant paths
cd "$(dirname "$0")"

dialogues="http://groups.inf.ed.ac.uk/maptask/signals/dialogues"
annotations="http://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip"
annotationDir=data
dialogueDir=data/dialogue-wavs

printf "${Yellow}"
echo "Download and extract annotations? (y/n)"
printf "${NC}"
read answer
if [[ $answer == 'y' || $answer == 'Y'   ]]; then
	printf "${Yellow}"
	echo "Downloading annotiations"
	echo "-----------------------"
	printf "${NC}"
	wget -P $annotationDir $annotations 
	cd $annotationDir
	unzip -qq hcrcmaptask.nxtformatv2-1.zip
	rm hcrcmaptask.nxtformatv2-1.zip
fi

printf "${Yellow}"
echo "Download dialogues(~3gb)? (y/n)"
printf "${NC}"
read answer
if [[ $answer == 'y' || $answer == 'Y'   ]]; then
	printf "${Yellow}"
	echo "Downloading Data ~3gb"
	echo "-----------------------"
	printf "${NC}"
	wget -P $dialogueDir -r -np -R "index.html*" -nd $dialogues
fi

printf "${GREEN}"
echo "DONE!"
printf "${NC}"
