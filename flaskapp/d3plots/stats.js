const meanRewards = {
    
    f_raw: (_data, window=null) => {
        let data = _data['results'];
        // take the average mean over window from the end
        if(window==null){
            window = data.length;
        }
        let meanRewardSum = 0;
        for(let i=0; i<data.length; i++){
            meanRewardSum += jStat.mean(data[i].slice(data[i].length-window,data[i].length));
        }
        return meanRewardSum/data.length;
    },
    f: (_data, scale, window=null) => {
        let data = _data['results'];
        // take the average mean over window from the end
        if(window==null){
            window = data.length;
        }
        let meanRewardSum = 0;
        for(let i=0; i<data.length; i++){
            meanRewardSum += jStat.mean(data[i].slice(data[i].length-window,data[i].length));
        }
        return scale(meanRewardSum/data.length);
    },
    label: (window) => "Average Mean Reward Over Last "+window+" Episodes",
    range: [150,700]
}

const stdRewards = {
    f_raw: (_data, window=null) => {
        let data = _data['results'];
        // take the average standard deviation over window from the end
        if(window==null){
            window = data.length;
        }
        let stdRewardSum = 0;
        for(let i=0; i<data.length; i++){
            stdRewardSum += jStat.stdev(data[i].slice(data[i].length-window,data[i].length));
        }
        return stdRewardSum/data.length;
    },
    f: (_data, scale, window=null) => {
        let data = _data['results'];
        // take the average standard deviation over window from the end
        if(window==null){
            window = data.length;
        }
        let stdRewardSum = 0;
        for(let i=0; i<data.length; i++){
            stdRewardSum += jStat.stdev(data[i].slice(data[i].length-window,data[i].length));
        }
        return scale(stdRewardSum/data.length);
    },
    label: (window) => "Average STD Reward Over Last "+window+" Episodes",
    range: [0,350]
}

const runningAvg = (totalrewards, window=100) =>{
  
  let running_avg = [totalrewards[0]];
  for(let i = 1; i< totalrewards.length; i++){
      running_avg.push(jStat.mean(totalrewards.slice(Math.max(0,i-window),i)));
  }
  return running_avg;

}

const engPower = {
    f_raw: (_data, window=null) => {
        let data = _data['design'];
  
        return data.eng_power;
    },
    f: (_data, scale, window=null) => {
        let data = _data['design'];
        // take the average mean over window from the end
        return scale(data.eng_power);
    },
    label: (window) => "Horsepower",
    range: [0,1000000]
}

const frictionLim = {
    f_raw: (_data, window=null) => {
        let data = _data['design'];
  
        return data.friction_lim;
    },
    f: (_data, scale, window=null) => {
        let data = _data['design'];
        // take the average mean over window from the end
        return scale(data.friction_lim);
    },
    label: (window) => "Tire Tread",
    range: [0,14000]
}

const maxSpeed = {
    f_raw: (_data, window=null) => {
        let data = _data['design'];
  
        return data.max_speed;
    },
    f: (_data, scale, window=null) => {
        let data = _data['design'];
        // take the average mean over window from the end
        return scale(data.max_speed);
    },
    label: (window) => "Max Speed Limit",
    range: [0,200]
}

const cost = {
    f_raw: (_data, window=null) => {
        let data = _data['design'];
  
        return calculateCarCost(data);
    },
    f: (_data, scale, window=null) => {
        let data = _data['design'];
        // take the average mean over window from the end
        return scale(calculateCarCost(data));
    },
    label: (window) => "Car Cost",
    range: [0,300000]

}