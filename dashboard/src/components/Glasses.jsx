import Image from "next/image";

const Glasses = ({ size = "md" }) => {
  const dimensions = {
    sm: { width: 250, height: 200 },
    md: { width: 400, height: 231 },
    lg: { width: 600, height: 347 },
  };
  
  const { width, height } = dimensions[size] || dimensions.md;
  
  return (
    <Image 
      src="/glasses.svg" 
      alt="Smart glasses" 
      width={width} 
      height={height} 
      priority
    />
  );
};

export default Glasses;
