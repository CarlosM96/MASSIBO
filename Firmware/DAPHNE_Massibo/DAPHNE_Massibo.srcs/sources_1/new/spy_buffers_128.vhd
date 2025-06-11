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
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;
use work.daphne_package.all;

entity spy_buffers_128 is
  Port ( 
    clka:  in std_logic;  
    reset: in std_logic; -- reset sync to clka
    trig:  in std_logic; -- trigger pulse sync to clka
    trig_sync: in std_logic;
    afe_dout_filtered:   in array_9x16_type; -- data bus from AFE channel

    clkb:  in  std_logic;
    addrb: in  std_logic_vector(11 downto 0);

    tx_rden: in std_logic;
    rx_addr_reg: in std_logic_vector(31 downto 0);
    
    fifo_2_data: out std_logic_vector(15 downto 0);

    
  
    --spy_bufr: out array_9x16_type
    spy_bufr_append: out std_logic_vector(143 downto 0)
    
  );
end spy_buffers_128;


architecture Behavioral of spy_buffers_128 is
signal spy_bufr: array_9x16_type;

--FIFO

signal link_ready, re, we, fifo_full, fifo_empty: std_logic;
signal write_clk : std_logic;
signal fifo_data : std_logic_vector(15 downto 0);
signal data_in : std_logic_vector(15 downto 0);
type state_type is (rst, wait4trig,readWaves, store, wait4done, fiforst,wait4fifotrig,readfifo,wait4donefifo );
signal state: state_type;

type state_type_fifo is (fiforst,wait4fifotrig,readfifo,wait4donefifo );
signal state_fifo: state_type_fifo;



signal wave_count : std_logic_vector(6 downto 0);
signal fifo_wr_addr: std_logic_vector(14 downto 0);
signal trig_fifo: std_logic;

signal dia_reg: std_logic_vector(15 downto 0);
signal spy_data: std_logic_vector(15 downto 0);

signal samples_counter: std_logic_vector(11 downto 0);

signal din0,din1,din2,trig_spy128: std_logic;

signal re_from_spy: std_logic;
signal fifo_in, fifo_out: std_logic_vector(15 downto 0);

signal re_from_spy_vec: std_logic_vector(8 downto 0);


signal fifo_RDEN: std_logic;

component spy_128 is
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
end component spy_128;


component FIFO16 is
    Port (
        link_ready : in  std_logic;
        reset      : in  std_logic;
        re         : in  std_logic;
        we         : in  std_logic;
        fifo_full  : out std_logic;
        fifo_empty : out std_logic;
        sysclk  : in  std_logic;
        tx_data    : out std_logic_vector(15 downto 0);
        wr_addr    : out std_logic_vector(14 downto 0);
        data_in    : in  std_logic_vector(15 downto 0)
    );
end component FIFO16;

component FIFO16_2 is
    Port (
        link_ready : in  std_logic;
        reset      : in  std_logic;
        re         : in  std_logic;
        we         : in  std_logic;
        fifo_full  : out std_logic;
        fifo_empty : out std_logic;
        mclk  : in  std_logic;
        oeiclk  : in  std_logic;
        tx_data    : out std_logic_vector(15 downto 0);
        wr_addr    : out std_logic_vector(14 downto 0);
        data_in    : in  std_logic_vector(15 downto 0)
    );
end component FIFO16_2;






begin



        gen_spy_bit: for b in 8 downto 0 generate
        
            spy_inst: spy_128
            port map(
                -- mclk domain
                clka  => clka,
                reset => reset,
                trig  => trig_sync,
                --dia   => afe_dout_filtered(b),
                dia => spy_data,
             
                -- oeiclk domain    
                clkb  => clkb,
                addrb => addrb,
                dob   => spy_bufr(b),
                we_reg_out=> re_from_spy_vec(b)
                
                
                );

        end generate gen_spy_bit;
        
        FIFO_0 : FIFO16
        port map (
            link_ready => '1',
            reset      => reset,
            --re         => re,
            re         => re_from_spy_vec(0),
            we         => we,
            fifo_full  => fifo_full,
            fifo_empty => fifo_empty,
            sysclk  => clka,
            tx_data    => fifo_data,
            wr_addr    =>fifo_wr_addr,
            data_in    => fifo_in
            --data_in    => spy_data
        );
        
        
        fifo_RDEN <= '1' when (std_match(rx_addr_reg,FIFO_TOP_ADDR) and tx_rden='1') else '0'; 
    
        FIFO : FIFO16_2
            port map (
                link_ready => '1',
                reset      => reset,
                re         => fifo_RDEN,
                we         => we,
                --fifo_full  => fifo_full,
                --fifo_empty => fifo_empty,
                mclk  => clka,
                oeiclk => clkb,
                tx_data    => fifo_2_data,
                wr_addr    =>open,
                data_in    => fifo_in
            );



    gen_spy_signals: for b in 8 downto 0 generate
            spy_bufr_append(((b)*16 + 15) downto ((b)*16)) <= spy_bufr(b);
    end generate gen_spy_signals;
    
    
           -- FSM to wait for trigger pulse and drive addr_reg (write pointer) and we_reg

    fifo_proc: process(clka)
    begin
        if rising_edge(clka) then
            if (fifo_wr_addr="001000000000000") then 
                trig_fifo <= '1';
            else
                trig_fifo <= '0';
            end if;
            
            if (reset='1') then
                wave_count <= "0000000";-------
                we   <= '0';
                state    <= rst;
                fifo_in <= "0000000000001111";
            else
                --dia_reg <= dia_delayed;
                fifo_in <= afe_dout_filtered(0);
                --output_selector => (others => '0'),
                --fifo_in <= "0000000000000000";

                case state is
                    when rst =>
                        state <= wait4trig;
                    when wait4trig =>
                        if (trig_sync='1') then -----trig
                            state <= store;
                            we <= '1';
                        else
                            state <= wait4trig;
                            we <= '0';
                        end if;
                        
                    when store =>
                  
                        if (fifo_full='1') then
                            state <= wait4done;
                            we <= '0';
                        
                        elsif (wave_count="1111111") then 
                            state <= readWaves;
                            we <= '0';
                            
                        else
                            
                            state <= store;
                            wave_count <= std_logic_vector(unsigned(wave_count) + 1);
                            we<= '1';
                            
                        end if;
                   
                    when readWaves =>
                        wave_count <= "0000000";
                        if (trig_sync='1') then ----trig
                            state <= store;
                            we <= '1';
                        else
                            state <= readWaves;
                            we <= '0';
                            --wave_count <= "0000000";
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
    end process fifo_proc;
    
    


--    fifo_read_proc: process(clka)
--    begin
--        if rising_edge(clka) then

--            if (reset='1') then
--                fifo_out  <= X"000F";
--                re   <= '0';
--                samples_counter <= X"000";
--                state_fifo    <= fiforst;
--            else
--                 fifo_out<= fifo_data;

--                case state_fifo is
--                    when fiforst =>
--                        state_fifo <= wait4fifotrig;
                        
--                    when wait4fifotrig =>
--                        if (trig_fifo='1') then
--                            state_fifo <= readfifo;
--                            re <= '1';
--                        else
--                            state_fifo <= wait4fifotrig;
--                            re <= '0';
--                            samples_counter <= X"000";
--                        end if;
                        
--                    when readfifo =>
                    
--                        if (samples_counter=X"FFF") then
--                            state_fifo <= wait4donefifo;
--                            re <= '0';
                        
--                        else
--                            state_fifo <= readfifo;
--                            samples_counter <= std_logic_vector(unsigned(samples_counter) + 1);
--                            re <= '1';
                            
--                        end if;
                   
                    
                        
--                    when wait4donefifo =>
--                        if (trig_fifo='0') then
--                            state_fifo <= wait4fifotrig;
--                        else
--                            state_fifo <= wait4donefifo;
--                        end if;
--                    when others => 
--                        state_fifo <= fiforst;    
    
--                end case;
--            end if;
--        end if;
--    end process fifo_read_proc;

    trig_delay: process(clka)
    begin
        if rising_edge(clka) then
            din0 <= trig_fifo;  -- latest sample
            din1 <= din0; -- previous sample
            din2 <= din1; -- previous previous sample
        end if;
    end process trig_delay;

    trig_spy128 <= (din0 or din1 or din2);
    
    
    
    
    data: process(clka)
    begin 
        if re_from_spy_vec(0)='1' then --re_from_spy
            spy_data<=fifo_data;
        else
            spy_data<="0001010110110011";     
        end if;
    end process data;

end Behavioral;
