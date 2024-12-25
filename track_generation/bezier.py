class Bezier:
    PASCAL = [
             [1],
            [1,1],          
           [1,2,1],         
          [1,3,3,1],        
         [1,4,6,4,1],       
        [1,5,10,10,5,1],    
       [1,6,15,20,15,6,1]] 
    
    def __init__(self, id: int, interval: float = 0.01) -> None:
        self.interval = interval
    
    # parametric curve
    # t varies from 0 to 1 in constant intervals
    def _binomial(self, n: int, k: int) -> int:
        while (n >= len(self.PASCAL)):
            rows = len(self.PASCAL)
            new_row = [1]            
            # add new row if not present
            prev = rows-1
            for i in range(1, rows):  
                new_row.append(self.PASCAL[prev][i-1] + self.PASCAL[prev][i])   
            new_row.append(1)
            self.PASCAL.append(new_row)
        return self.PASCAL[n][k]

    def _cubic_bezier(t: float) -> float:
        mt = 1-t
        return (mt^3) + (3 * mt^2*t) + (3 * mt*t^2) + t^3
    
    def _n_bezer(n: int, t: float) -> float:
        
        return 
              
if __name__ == '__main__':
    bezier = Bezier(1)
    print(bezier._binomial(10, 1))