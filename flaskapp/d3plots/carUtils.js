const calculateCarCost = (config)=>{
    //body material cost
    const bodyCost = calculateBodyCost(config);
    //tire cost
    const tireCost = calculateTireCost(config);
    //wheel cost
    const wheelCost = calculateWheelCost(config);
    //engine cost
    const engineCost = calculateEngineCost(config);
    //drivetrain cost
    const drivetrainCost = calculateDrivetrainCost(config);
    return bodyCost + tireCost + wheelCost + engineCost + drivetrainCost;
  }
  
const calculateBodyCost = (config)=>{
    //get the area of each hull and sum for now
    let materialCost = 10; //cost per pixel of material
    //in the future, this will be a function^^
    return materialCost*calculateBodyWeight(config);
  }
const calculateTireCost = (config) =>{
    //tire cost is based on the width, tread, and wheel radius
    return 8*(config.wheel_rad+config.wheel_width+config.friction_lim/2);
  }
const calculateWheelCost = (config)=>{
    //wheel cost is based on wheel radius and moment
    return 4*(Math.PI*config.wheel_rad*config.wheel_rad)+10*config.wheel_width;
  }
   const calculateEngineCost = (config)=>{
    //engine cost is based on power
    return 25*(config.eng_power/1e4)*(config.eng_power/1e4);
  }
   const calculateDrivetrainCost = (config)=>{
    //switch statement for fwd, rwd, awd
    return 1000;//
  }
  
   const trapezoidArea = (vertices)=>{
    let yMax = vertices.reduce((a,v)=>Math.max(a,v[1]),-1e7);
    let yMin = vertices.reduce((a,v)=>Math.min(a,v[1]),yMax);
    let h = Math.abs(yMax-yMin);
    let b1verts = vertices.filter((v)=>v[1]==yMax);
    let b2verts = vertices.filter((v)=>v[1]==yMin);
    let b1 = Math.abs(b1verts[0][0]-b1verts[1][0]);
    let b2 = Math.abs(b2verts[0][0]-b2verts[1][0]);
    return 0.5*(b1+b2)*h;
  }
  
   const octagonArea = (vertices)=>{
    let yMax = vertices.reduce((a,v)=>Math.max(a,v[1]),-1e7);
    let yMin = vertices.reduce((a,v)=>Math.min(a,v[1]),yMax);
    let t1verts = vertices.filter((v)=>v[1]==yMax);
    let t2verts = vertices.filter((v)=>v[1]==yMin);
    let centerverts = vertices.filter((v)=>(v[1]!=yMax && v[1]!=yMin));
    let centerYMax = centerverts.reduce((a,v)=>Math.max(a,v[1]),-1e7);
    let centerYMin = centerverts.reduce((a,v)=>Math.min(a,v[1]),centerYMax);
    t1verts = t1verts.concat(centerverts.filter(v=>v[1]==centerYMax));
    t2verts = t2verts.concat(centerverts.filter(v=>v[1]==centerYMin));
    // console.log("CENTERVERTS",centerverts,"T1VERTS",t1verts,"CENTERYMAX",centerYMax);
    return trapezoidArea(t1verts)+trapezoidArea(t2verts)+trapezoidArea(centerverts);
  }
  
   const calculateBodyWeight = (config)=>{
    let hulls = [config.hull_poly1,config.hull_poly2,config.hull_poly3,config.hull_poly4];
    let areas = [];
    for(let hull in hulls){
      if(hulls[hull].length == 4){
        areas.push(trapezoidArea(hulls[hull]));
      }
      else if(hulls[hull].length == 8){
        areas.push(octagonArea(hulls[hull]));
      }
      else{
        console.log("unknown polygon");
      }
    }
    // console.log("ARRRRIAAAAAS",areas)
    return 0.125*areas.map((h,i)=>h*config.hull_densities[i]).reduce((a,h)=>a+h);
  }
  
   const calculateEngineWeight = (config)=>{
    //for now, 1hp=1kg
    return config.eng_power/1e3;
  }
  
   const calculateCarWeight = (config)=>{
    let bodyWeight = calculateBodyWeight(config);
    let engineWeight = calculateEngineWeight(config);
    // console.log("WEIGHTS:",bodyWeight,engineWeight);
    return bodyWeight+engineWeight;
  }
  
   const calculateCarLength = (config)=>{
    let bumper = config.hull_poly1;
    let spoiler = config.hull_poly4;
    let tip = bumper.reduce((a,p)=>Math.max(a,p[1]),0);//need a really large number
    let tail = spoiler.reduce((a,p)=>Math.min(a,p[1]),1e5);
    return Math.abs(tip-tail);
  }
  
   const calculateCarWidth = (config) => {
    let max=0;
    let hulls = [config.hull_poly1,config.hull_poly2,config.hull_poly3,config.hull_poly4];
    for(let hull in hulls){
      let maxX = hulls[hull].reduce((a,p)=>Math.max(a,p[0]),0);
      if(maxX > max){
        max=maxX;
      }
    }
  
    let min=max;
    for(let hull in hulls){
      let minX = hulls[hull].reduce((a,p)=>Math.min(a,p[0]),max);
      if(minX < min){
        min=minX;
      }
    }
    return Math.abs(max-min);
  }
  