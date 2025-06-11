----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/19/2024 04:30:56 PM
-- Design Name: 
-- Module Name: spy_buffers - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;
use work.daphne_package.all;

entity spy_buffers is
  Port ( 
    clka:  in std_logic;  
    reset: in std_logic; -- reset sync to clka
    trig:  in std_logic; -- trigger pulse sync to clka
    afe_dout_filtered:   in array_9x16_type; -- data bus from AFE channel

    clkb:  in  std_logic;
    addrb: in  std_logic_vector(11 downto 0);

  
    --spy_bufr: out array_9x16_type
    spy_bufr_append: out std_logic_vector(143 downto 0)
  );
end spy_buffers;


architecture Behavioral of spy_buffers is
signal spy_bufr: array_9x16_type;



component spy is
port(

    clka:  in std_logic;  
    reset: in std_logic; -- reset sync to clka
    trig:  in std_logic; -- trigger pulse sync to clka
    dia:   in std_logic_vector(15 downto 0); -- data bus from AFE channel

    clkb:  in  std_logic;
    addrb: in  std_logic_vector(11 downto 0);
    dob:   out std_logic_vector(15 downto 0)

  );
end component spy;




begin



        gen_spy_bit: for b in 8 downto 0 generate
        
            spy_inst: spy
            port map(
                -- mclk domain
                clka  => clka,
                reset => reset,
                trig  => trig,
                dia   => afe_dout_filtered(b),
                -- oeiclk domain    
                clkb  => clkb,
                addrb => addrb,
                dob   => spy_bufr(b));

        end generate gen_spy_bit;

    gen_spy_signals: for b in 8 downto 0 generate
            spy_bufr_append(((b)*16 + 15) downto ((b)*16)) <= spy_bufr(b);
    end generate gen_spy_signals;

end Behavioral;
