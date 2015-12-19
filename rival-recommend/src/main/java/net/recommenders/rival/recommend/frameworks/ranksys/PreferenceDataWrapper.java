/*
 * Copyright 2015 recommenders.net.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.recommenders.rival.recommend.frameworks.ranksys;

import es.uam.eps.ir.ranksys.core.preference.IdPref;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.AbstractFastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.IntIterator;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import net.recommenders.rival.core.DataModel;
import org.ranksys.core.util.iterators.StreamDoubleIterator;
import org.ranksys.core.util.iterators.StreamIntIterator;

/**
 * RankSys' FastPreferenceData wrapper for {@link net.recommenders.rival.core.DataModel}.
 *
 * @author Saúl Vargas (Saul.Vargas@mendeley.com)
 */
public class PreferenceDataWrapper<U, I> extends AbstractFastPreferenceData<U, I> {

    private final DataModel<U, I> dataModel;

    public PreferenceDataWrapper(DataModel<U, I> dataModel, FastUserIndex<U> users, FastItemIndex<I> items) {
        super(users, items);

        this.dataModel = dataModel;
    }

    @Override
    public int numUsersWithPreferences() {
        return dataModel.getNumUsers();
    }

    @Override
    public int numItemsWithPreferences() {
        return dataModel.getNumItems();
    }

    @Override
    public int numUsers(I i) {
        return dataModel.getItemUserPreferences().get(i).size();
    }

    @Override
    public int numItems(U u) {
        return dataModel.getUserItemPreferences().get(u).size();
    }

    @Override
    public int numPreferences() {
        return dataModel.getUserItemPreferences().values().stream()
                .mapToInt(m -> m.size())
                .sum();
    }

    @Override
    public Stream<U> getUsersWithPreferences() {
        return dataModel.getUsers().stream();
    }

    @Override
    public Stream<I> getItemsWithPreferences() {
        return dataModel.getItems().stream();
    }

    @Override
    public Stream<? extends IdPref<I>> getUserPreferences(U u) {
        return dataModel.getUserItemPreferences().get(u).entrySet().stream()
                .map(e -> new IdPref<I>(e.getKey(), e.getValue()));
    }

    @Override
    public Stream<? extends IdPref<U>> getItemPreferences(I i) {
        return dataModel.getItemUserPreferences().get(i).entrySet().stream()
                .map(e -> new IdPref<U>(e.getKey(), e.getValue()));
    }

    @Override
    public int numUsers(int iidx) {
        return numUsers(iidx2item(iidx));
    }

    @Override
    public int numItems(int uidx) {
        return numItems(uidx2user(uidx));
    }

    @Override
    public IntStream getUidxWithPreferences() {
        return getUsersWithPreferences().mapToInt(this::user2uidx);
    }

    @Override
    public IntStream getIidxWithPreferences() {
        return getItemsWithPreferences().mapToInt(this::item2iidx);
    }

    @Override
    public Stream<? extends IdxPref> getUidxPreferences(int uidx) {
        return getUserPreferences(uidx2user(uidx))
                .map(p -> new IdxPref(item2iidx(p.id), p.v));
    }

    @Override
    public Stream<? extends IdxPref> getIidxPreferences(int iidx) {
        return getItemPreferences(iidx2item(iidx))
                .map(p -> new IdxPref(user2uidx(p.id), p.v));
    }

    @Override
    public IntIterator getUidxIidxs(int uidx) {
        return new StreamIntIterator(getUidxPreferences(uidx)
                .mapToInt(p -> p.idx));
    }

    @Override
    public DoubleIterator getUidxVs(int uidx) {
        return new StreamDoubleIterator(getIidxPreferences(uidx)
                .mapToDouble(p -> p.v));
    }

    @Override
    public IntIterator getIidxUidxs(int iidx) {
        return new StreamIntIterator(getIidxPreferences(iidx)
                .mapToInt(p -> p.idx));
    }

    @Override
    public DoubleIterator getIidxVs(int iidx) {
        return new StreamDoubleIterator(getIidxPreferences(iidx)
                .mapToDouble(p -> p.v));
    }

    @Override
    public boolean useIteratorsPreferentially() {
        return false;
    }

}
