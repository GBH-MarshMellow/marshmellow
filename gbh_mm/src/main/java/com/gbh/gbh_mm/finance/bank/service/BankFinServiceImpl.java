package com.gbh.gbh_mm.finance.bank.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.BankAPI;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class BankFinServiceImpl implements BankFinService {
    private final BankAPI bankAPI;

    @Autowired
    public BankFinServiceImpl(BankAPI bankAPI) {
        this.bankAPI = bankAPI;
    }

    @Override
    public Map<String, Object> findBankList() throws JsonProcessingException {
        Map<String, Object> result = bankAPI.findBankList();
        return result;

    }
}
