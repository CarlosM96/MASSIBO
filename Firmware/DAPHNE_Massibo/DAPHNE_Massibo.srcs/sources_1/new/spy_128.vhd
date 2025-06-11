-- spy.vhd
-- DAPHNE spy buffer for one link AFE output, 4k deep
-- reset only resets the trigger logic, it does not erase the buffer contents
-- triggered on low to to high pulse, then writes next 4096 words into buffer, stops and waits for next trigger
-- add delay elements to produce 64 deep delay for fixed pretrigger
-- Jamieson Olsen <jamieson@fnal.gov>

-- port A = 4k x 16 (write) clock domain from afe front end
-- port B = 4k x 16 (read)  connected to gigabit ethernet interface

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

use work.daphne_package.all;

entity spy_128 is
port(

    clka:  in std_logic;  
    reset: in std_logic; -- reset sync to clka
    trig:  in std_logic; -- trigger pulse sync to clka
    dia:   in std_logic_vector(15 downto 0); -- data bus from AFE channel

    clkb:  in  std_logic;
    addrb: in  std_logic_vector(11 downto 0);
    dob:   out std_logic_vector(15 downto 0);
    
    we_reg_out: out std_logic

  );
end spy_128;

architecture spy_arch of spy_128 is

    signal addr_reg: std_logic_vector(11 downto 0);
    signal wave_count : std_logic_vector(6 downto 0);
    signal dia_reg, dia_q0, dia_q1, dia_q2, dia_q3, dia_q4, dia_q5, dia_q6, dia_q7, dia_q8, dia_delayed:  std_logic_vector(15 downto 0);
    signal dia_q9, dia_q10, dia_q11, dia_q12, dia_q13, dia_q14, dia_q15, dia_q16:  std_logic_vector(15 downto 0);
    signal we_reg:   std_logic;

    type state_type is (rst, wait4trig,readWaves, store, wait4done);
    signal state: state_type;

begin

     --input delay elements for fixed pre-trigger cascade two 32 bit shift registers

     --input delay elements for fixed pre-trigger cascade two 32 bit shift registers

--    gendelay: for i in 15 downto 0 generate

--        srlc32e_0_inst : srlc32e
--        port map(
--            clk => clka,
--            ce => '1',
--            a => "11111",
--            d => dia(i),
--            q => open,
--            q31 => dia_q0(i)  
--        );
        
--        srlc32e_1_inst : srlc32e
--        port map(
--            clk => clka,
--            ce => '1',
--            a => "11111",
--            d => dia_q0(i),
--            q => dia_q1(i),
--            q31 => open  
--        );
        
--        srlc32e_2_inst : srlc32e
--        port map(
--            clk => clka,
--            ce => '1',
--            a => "11111",
--            d => dia_q1(i),
--            q => open,
--            q31 => dia_q2(i)  
--        );
        
--        srlc32e_3_inst : srlc32e
--        port map(
--            clk => clka,
--            ce => '1',
--            a => "11111",
--            d => dia_q2(i),
--            q => open,
--            q31 => dia_q3(i)  
--        );
        
--        srlc32e_4_inst : srlc32e
--        port map(
--            clk => clka,
--            ce => '1',
--            a => "11111",
--            d => dia_q3(i),
--            q => open,
--            q31 => dia_q4(i)   --original
--        );
        
--        srlc32e_n_inst : srlc32e
--        port map(
--            clk => clka,
--            ce => '1',
--            a => "11111",
--            d => dia_q4(i),
--            q => dia_delayed(i),
--            q31 => open  
--        );

--    end generate gendelay;





    -- FSM to wait for trigger pulse and drive addr_reg (write pointer) and we_reg

--    fsm_proc: process(clka)
--    begin
--        if rising_edge(clka) then

--            if (reset='1') then
--                wave_count <= "0000000";-------
--                dia_reg  <= X"0000";
--                we_reg   <= '0';
--                addr_reg <= X"000";
--                state    <= rst;
--            else
--                --dia_reg <= dia_delayed;
--                dia_reg <= dia;

--                case state is
--                    when rst =>
--                        state <= wait4trig;
--                    when wait4trig =>
--                        if (trig='1') then
--                            state <= store;
--                            we_reg <= '1';
--                        else
--                            state <= wait4trig;
--                            we_reg <= '0';
--                            addr_reg <= X"000";
--                        end if;
                        
--                    when store =>
                    
--                        if (addr_reg=X"FFF") then
--                            state <= wait4done;
--                            we_reg <= '0';
                        
--                        elsif (wave_count="1111111") then 
--                            state <= readWaves;
--                            we_reg <= '0';
                            
--                        else
                            
--                            state <= store;
--                            addr_reg <= std_logic_vector(unsigned(addr_reg) + 1);
--                            wave_count <= std_logic_vector(unsigned(wave_count) + 1);
--                            we_reg <= '1';
                            
--                        end if;
                   
--                    when readWaves =>
--                        wave_count <= "0000000";
--                        if (trig='1') then
--                            state <= store;
--                            we_reg <= '1';
--                        else
--                            state <= readWaves;
--                            we_reg <= '0';
--                            --wave_count <= "0000000";
--                        end if;
                        
                        
                        
--                    when wait4done =>
--                        if (trig='0') then
--                            state <= wait4trig;
--                        else
--                            state <= wait4done;
--                        end if;
                        
--                    when others => 
--                        state <= rst;    
    
--                end case;
--            end if;
--        end if;
--    end process fsm_proc;
    
    
    
        fsm_proc: process(clka)
    begin
        if rising_edge(clka) then

            if (reset='1') then
                dia_reg  <= X"0000";
                we_reg   <= '0';
                addr_reg <= X"000";
                state    <= rst;
            else
                dia_reg <= dia;

                case state is
                    when rst =>
                        state <= wait4trig;
                    when wait4trig =>
                        if (trig='1') then
                            state <= store;
                            we_reg <= '1';
                        else
                            state <= wait4trig;
                            we_reg <= '0';
                            addr_reg <= X"000";
                        end if;
                        
                    when store =>
                    
                        if (addr_reg=X"FFF") then
                            state <= wait4done;
                            we_reg <= '0';
                        
                        else
                            state <= store;
                            addr_reg <= std_logic_vector(unsigned(addr_reg) + 1);
                            we_reg <= '1';
                            
                        end if;
                   
                    
                        
                    when wait4done =>
                        if (trig='0') then
                            state <= wait4trig;
                        else
                            state <= wait4done;
                        end if;
                    when others => 
                        state <= rst;    
    
                end case;
            end if;
        end if;
    end process fsm_proc;

    -- instantiate four 18k true dual port blockrams to create a single 4k x 16 buffer
    -- port A driven by the trigger logic in this module
    -- port B is read only, controlled by the GbE interface

    genbuffer: for i in 3 downto 0 generate

        ramb_inst: ramb16_s4_s4 -- TDP blockram "older style" primitive both ports are 4k x 4
        generic map (
            SIM_COLLISION_CHECK => "NONE",
            INIT_A => X"0",
            INIT_B => X"0",
            SRVAL_A => X"0",
            SRVAL_B => X"0",
            WRITE_MODE_A => "WRITE_FIRST",
            WRITE_MODE_B => "WRITE_FIRST"
        )
        port map(
             clka  => clka,
             addra => addr_reg(11 downto 0),
             dia   => dia_reg( ((i*4)+3) downto (i*4) ),
             ena   => we_reg,
             ssra  => '0',
             wea   => we_reg,
    
             clkb  => clkb,
             addrb => addrb(11 downto 0),
             dib   => "0000",
             enb   => '1',
             ssrb  => '0',
             web   => '0',
             dob   => dob( ((i*4)+3) downto (i*4) )
            );

    end generate;
    
    we_reg_out <= we_reg;

end spy_arch;
